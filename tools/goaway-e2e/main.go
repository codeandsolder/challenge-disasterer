package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"

	"git.gammaspectra.live/git/go-away/lib/challenge/wasm"
	_interface "git.gammaspectra.live/git/go-away/lib/challenge/wasm/interface"
	"github.com/tetratelabs/wazero/api"
)

type challengeConfig struct {
	Challenge  string `json:"challenge"`
	Difficulty uint64 `json:"difficulty"`
}

type solveResponse struct {
	SolveMS        float64 `json:"solve_ms"`
	Implementation string  `json:"implementation"`
	Answer         struct {
		Token string `json:"token"`
		Nonce string `json:"nonce"`
		ID    string `json:"id"`
	} `json:"answer"`
}

type solverCapability struct {
	Solver          string   `json:"solver"`
	DifficultyMin   uint64   `json:"difficulty_min"`
	DifficultyMax   uint64   `json:"difficulty_max"`
	Implementations []string `json:"implementations"`
}

type statsResponse struct {
	SolverCapabilities []solverCapability `json:"solver_capabilities"`
}

func contains(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func assertCapabilities(serviceURL, implementation string) error {
	response, err := http.Get(serviceURL + "/stats")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("stats status=%d", response.StatusCode)
	}
	var stats statsResponse
	if err := json.NewDecoder(response.Body).Decode(&stats); err != nil {
		return err
	}
	for i := range stats.SolverCapabilities {
		row := &stats.SolverCapabilities[i]
		if row.Solver == "goaway" && row.DifficultyMin == 0 && row.DifficultyMax == 256 {
			if contains(row.Implementations, "auto") && contains(row.Implementations, "safe") && contains(row.Implementations, implementation) {
				return nil
			}
			return fmt.Errorf("go-away full-domain capability omits implementation %q: %+v", implementation, row)
		}
	}
	return fmt.Errorf("missing go-away 0..256 full-domain capability")
}

func main() {
	if len(os.Args) != 3 {
		panic("usage: goaway-e2e RUNTIME_WASM SERVICE_URL")
	}
	wasmData, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	serviceURL := os.Args[2]

	runner := wasm.NewRunner(true)
	defer runner.Close()
	if err := runner.Compile("goaway-e2e", wasmData); err != nil {
		panic(err)
	}
	implementation := os.Getenv("CHALLENGE_IMPLEMENTATION")
	if implementation == "" {
		implementation = "auto"
	}
	if err := assertCapabilities(serviceURL, implementation); err != nil {
		panic(err)
	}

	difficulties := []string{"0", "8", "13", "20"}
	for _, difficulty := range difficulties {
		if err := runCase(runner, serviceURL, difficulty); err != nil {
			panic(err)
		}
	}
}

func runCase(runner *wasm.Runner, serviceURL, difficulty string) error {
	key := []byte("challenge-disasterer-goaway-e2e-key")
	parameters := map[string]string{"difficulty": difficulty}

	var config challengeConfig
	err := runner.Instantiate("goaway-e2e", func(ctx context.Context, mod api.Module) error {
		out, err := wasm.MakeChallengeCall(ctx, mod, _interface.MakeChallengeInput{
			Key:        key,
			Parameters: parameters,
		})
		if err != nil {
			return err
		}
		if out.Code != http.StatusOK {
			return fmt.Errorf("MakeChallenge code=%d error=%s", out.Code, out.Error)
		}
		return json.Unmarshal(out.Data, &config)
	})
	if err != nil {
		return fmt.Errorf("official MakeChallenge: %w", err)
	}

	implementation := os.Getenv("CHALLENGE_IMPLEMENTATION")
	if implementation == "" {
		implementation = "auto"
	}
	payload, _ := json.Marshal(map[string]any{
		"solver":         "goaway",
		"implementation": implementation,
		"challenge":      config,
	})
	response, err := http.Post(serviceURL+"/solve", "application/json", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("solver status=%d body=%s", response.StatusCode, body)
	}
	var solved solveResponse
	if err := json.Unmarshal(body, &solved); err != nil {
		return err
	}
	if solved.Implementation != implementation {
		return fmt.Errorf("requested implementation %s, got %s", implementation, solved.Implementation)
	}

	// The upstream verifier checks the token. Independently check that the
	// response metadata carries the canonical SHA-256 digest prefix, too.
	nonce, err := strconv.ParseUint(solved.Answer.Nonce, 10, 64)
	if err != nil {
		return fmt.Errorf("invalid go-away decimal nonce %q: %w", solved.Answer.Nonce, err)
	}
	if expectedToken := fmt.Sprintf("%s%016x", config.Challenge, nonce); solved.Answer.Token != expectedToken {
		return fmt.Errorf("go-away token does not match challenge and nonce")
	}
	challengeBytes, err := hex.DecodeString(config.Challenge)
	if err != nil {
		return fmt.Errorf("go-away challenge is not hexadecimal: %w", err)
	}
	if len(challengeBytes) != 32 {
		return fmt.Errorf("go-away challenge is %d bytes, expected 32", len(challengeBytes))
	}
	var nonceBytes [8]byte
	binary.BigEndian.PutUint64(nonceBytes[:], nonce)
	digest := sha256.Sum256(append(challengeBytes, nonceBytes[:]...))
	if expectedID := hex.EncodeToString(digest[:16]); solved.Answer.ID != expectedID {
		return fmt.Errorf("go-away digest ID mismatch: got %q, expected %q", solved.Answer.ID, expectedID)
	}

	verify := func(result []byte) (_interface.VerifyChallengeOutput, error) {
		var output _interface.VerifyChallengeOutput
		err := runner.Instantiate("goaway-e2e", func(ctx context.Context, mod api.Module) error {
			var err error
			output, err = wasm.VerifyChallengeCall(ctx, mod, _interface.VerifyChallengeInput{
				Key:        key,
				Parameters: parameters,
				Result:     result,
			})
			return err
		})
		return output, err
	}

	got, err := verify([]byte(solved.Answer.Token))
	if err != nil {
		return err
	}
	if got != _interface.VerifyChallengeOutputOK {
		return fmt.Errorf("official verifier rejected solution: %d", got)
	}

	corrupted := []byte(solved.Answer.Token)
	corruptIndex := len(corrupted) - 1
	if difficulty == "0" {
		// Every nonce is valid at difficulty zero, so changing only the nonce
		// cannot be a negative test. Corrupt the challenge binding instead.
		corruptIndex = 0
	}
	if corrupted[corruptIndex] == '0' {
		corrupted[corruptIndex] = '1'
	} else {
		corrupted[corruptIndex] = '0'
	}
	got, err = verify(corrupted)
	if err != nil {
		return err
	}
	if got == _interface.VerifyChallengeOutputOK {
		return fmt.Errorf("official verifier accepted corrupted solution")
	}

	fmt.Printf("go-away: PASS difficulty=%s solve_ms=%.3f\n", difficulty, solved.SolveMS)
	return nil
}
