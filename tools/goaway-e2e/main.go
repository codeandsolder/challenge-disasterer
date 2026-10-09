package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"git.gammaspectra.live/git/go-away/lib/challenge/wasm"
	_interface "git.gammaspectra.live/git/go-away/lib/challenge/wasm/interface"
	"github.com/tetratelabs/wazero/api"
)

type challengeConfig struct {
	Challenge  string `json:"challenge"`
	Difficulty uint8  `json:"difficulty"`
}

type solveResponse struct {
	SolveMS float64 `json:"solve_ms"`
	Answer  struct {
		Token string `json:"token"`
	} `json:"answer"`
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

	for _, difficulty := range []string{"8", "13", "20"} {
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

	payload, _ := json.Marshal(map[string]any{
		"solver":    "goaway",
		"challenge": config,
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
	if corrupted[len(corrupted)-1] == '0' {
		corrupted[len(corrupted)-1] = '1'
	} else {
		corrupted[len(corrupted)-1] = '0'
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
