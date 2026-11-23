mod load;
mod records;
mod summary;

use std::{io, path::PathBuf};

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Compare hardware-matrix run directories by exact benchmark keys")]
struct Args {
    #[arg(required = true)]
    runs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut collector = summary::Collector::default();
    for run in args.runs {
        collector.add(load::run(&run)?);
    }
    let result = collector.finish()?;
    serde_json::to_writer_pretty(io::stdout().lock(), &result)?;
    println!();
    Ok(())
}
