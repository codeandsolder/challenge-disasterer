//! Own every subprocess group until it has been terminated and reaped.
use std::{
    fs::File,
    io::{Read, Seek},
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use rustix::{
    io::Errno,
    process::{
        Pid, Signal, WaitId, WaitIdOptions, WaitOptions, kill_process_group, waitid, waitpgid,
    },
};

const INTERVAL: Duration = Duration::from_millis(50);

pub struct Supervisor {
    cancelled: Arc<AtomicBool>,
    command_timeout: Duration,
}

impl Supervisor {
    pub fn new(command_timeout: Duration) -> Result<Self> {
        let cancelled = Arc::new(AtomicBool::new(false));
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            signal_hook::flag::register(signal, Arc::clone(&cancelled))?;
        }
        rustix::process::set_child_subreaper(Some(Pid::INIT))?;
        Ok(Self {
            cancelled,
            command_timeout,
        })
    }

    pub fn check(&self, started: Instant, timeout: Duration) -> Result<()> {
        ensure!(
            !self.cancelled.load(Ordering::Relaxed),
            "fixture run cancelled"
        );
        ensure!(
            started.elapsed() < timeout,
            "fixture deadline exceeded ({timeout:?})"
        );
        Ok(())
    }

    pub fn run(&self, command: &mut Command) -> Result<()> {
        eprintln!("+ {command:?}");
        let mut child = GroupChild::spawn(command)?;
        let started = Instant::now();
        loop {
            if let Some(status) = child.poll()? {
                ensure!(status.success(), "{command:?} exited with {status}");
                return Ok(());
            }
            self.check(started, self.command_timeout)
                .with_context(|| format!("{command:?}"))?;
            thread::sleep(INTERVAL);
        }
    }

    pub fn capture(&self, command: &mut Command) -> Result<String> {
        let mut output = tempfile::tempfile()?;
        self.run(command.stdout(output.try_clone()?))?;
        output.rewind()?;
        let mut result = String::new();
        output.read_to_string(&mut result)?;
        Ok(result)
    }
}

pub struct GroupChild {
    child: Option<Child>,
    group: Pid,
}

impl GroupChild {
    pub fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .process_group(0)
            .spawn()
            .with_context(|| format!("spawn {command:?}"))?;
        let group = Pid::from_child(&child);
        Ok(Self {
            child: Some(child),
            group,
        })
    }

    pub fn logged(command: &mut Command, path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Self::spawn(
            command
                .stdout(file.try_clone()?)
                .stderr(file)
                .stdin(Stdio::null()),
        )
    }

    pub fn poll(&mut self) -> Result<Option<ExitStatus>> {
        // Observe exit without reaping the leader: its PGID cannot be reused
        // before we signal the entire group, including orphaned grandchildren.
        match waitid(
            WaitId::Pid(self.group),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
        ) {
            Ok(Some(_)) => self.terminate().map(Some),
            Ok(None) | Err(Errno::INTR) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn terminate(&mut self) -> Result<ExitStatus> {
        let child = self.child.as_mut().context("process already reaped")?;
        match kill_process_group(self.group, Signal::KILL) {
            Ok(()) | Err(Errno::SRCH) => {}
            Err(error) => return Err(error.into()),
        }
        let status = child.wait()?;
        self.child = None;
        loop {
            match waitpgid(self.group, WaitOptions::empty()) {
                Err(Errno::CHILD) => break,
                Ok(_) | Err(Errno::INTR) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(status)
    }
}

impl Drop for GroupChild {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.terminate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustix::process::test_kill_process;
    use std::fs;

    #[test]
    fn leader_exit_reaps_background_descendants() -> Result<()> {
        rustix::process::set_child_subreaper(Some(Pid::INIT))?;
        let directory = tempfile::tempdir()?;
        let pid_file = directory.path().join("pid");
        let mut child = GroupChild::spawn(
            Command::new("sh")
                .args(["-c", "sleep 60 & echo $! > \"$1\"; exit 7", "fixture"])
                .arg(&pid_file),
        )?;
        let started = Instant::now();
        loop {
            if let Some(status) = child.poll()? {
                assert_eq!(status.code(), Some(7));
                break;
            }
            ensure!(
                started.elapsed() < Duration::from_secs(5),
                "child never exited"
            );
            thread::sleep(INTERVAL);
        }
        let raw = fs::read_to_string(pid_file)?.trim().parse()?;
        let pid = Pid::from_raw(raw).context("invalid test pid")?;
        assert_eq!(test_kill_process(pid), Err(Errno::SRCH));
        Ok(())
    }

    #[test]
    fn timeout_stops_running_command() -> Result<()> {
        let supervisor = Supervisor::new(Duration::from_millis(100))?;
        let started = Instant::now();
        let result = supervisor.run(Command::new("sleep").arg("60"));
        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(5));
        Ok(())
    }
}
