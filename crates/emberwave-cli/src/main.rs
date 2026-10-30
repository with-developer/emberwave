//! Command line front end of emberwave: decode PyroWave (`.pwv1`) files with
//! wgpu, encode them, measure the decode and encode times, and inspect files.

/// `println!` that returns an error rather than panicking when standard
/// output is gone, as when it goes to `head`, which `main` then ends quietly.
macro_rules! out {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        writeln!(std::io::stdout(), $($arg)*)?
    }};
}

/// `eprintln!` that goes on when standard error is gone, as when it goes to
/// `head`, rather than panicking: what is not read is not missed.
macro_rules! err {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

mod bench;
mod bench_encode;
mod compare;
mod decode;
mod encode;
mod info;
mod quality;
mod stats;
mod y4m;

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use emberwave_core::bitstream::{ChromaSiting, SequenceHeader, YcbcrRange};
use emberwave_core::color::{ColorMatrix, ColorParams};
use emberwave_core::gpu::{GpuContext, GpuOptions, parse_backends};

/// The error type of the command line tool: anything that can be printed.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The bytes of the file at `path`, or an error that names it.
pub fn read_file(path: &std::path::Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()).into())
}

#[derive(Debug, Parser)]
#[command(
    name = "emberwave-cli",
    version,
    about = "Decode PyroWave (.pwv1) files with wgpu, and encode them"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Decode a file to PNG images or a Y4M video
    Decode(decode::Args),
    /// Encode a Y4M video to a file, each frame within a byte budget
    Encode(encode::Args),
    /// Measure decode time per frame
    Bench(bench::Args),
    /// Measure encode time per frame
    BenchEncode(bench_encode::Args),
    /// Print the header and frame statistics of a file
    Info(info::Args),
    /// Compare two Y4M files plane by plane (PSNR)
    Compare(compare::Args),
    /// Measure a stream against the pictures it was encoded from: PSNR and
    /// the error the encoder's rate control minimises
    Quality(quality::Args),
}

/// The graphics backend to run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    /// Whatever the machine has, or what the `EMBERWAVE_BACKEND` variable names
    Auto,
    /// Vulkan
    Vulkan,
    /// Metal
    Metal,
    /// Direct3D 12
    Dx12,
}

/// How to pick the GPU, shared by the commands that decode.
#[derive(Debug, Args)]
pub struct GpuArgs {
    /// Graphics backend
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,

    /// Insist on a software adapter (lavapipe, WARP)
    #[arg(long)]
    pub software: bool,
}

impl GpuArgs {
    /// Creates the GPU context, printing which adapter was chosen to stderr.
    pub fn create_context(&self, timestamps: bool) -> Result<GpuContext> {
        // An EMBERWAVE_BACKEND that names no backend is an error here.
        let mut options = GpuOptions::from_env()?;
        let name = match self.backend {
            Backend::Auto => None,
            Backend::Vulkan => Some("vulkan"),
            Backend::Metal => Some("metal"),
            Backend::Dx12 => Some("dx12"),
        };
        if let Some(backends) = name.and_then(parse_backends) {
            options.backends = backends;
        }
        options.force_fallback_adapter = self.software;
        options.timestamps = timestamps;

        let context = GpuContext::new(&options)?;
        let info = context.adapter_info();
        err!(
            "adapter: {} ({:?}, {:?})",
            info.name,
            info.backend,
            info.device_type
        );
        Ok(context)
    }
}

/// Matrix coefficients for the RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MatrixChoice {
    /// Take them from the file
    Auto,
    /// ITU-R BT.601
    Bt601,
    /// ITU-R BT.709
    Bt709,
    /// ITU-R BT.2020 (non-constant luminance)
    Bt2020,
}

/// Value range of the YUV planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RangeChoice {
    /// Take it from the file
    Auto,
    /// Full range, 0 to 255
    Full,
    /// Video range, 16 to 235 (240 for chroma)
    Limited,
}

/// How to convert YUV to RGB, shared by the commands that produce RGB.
#[derive(Debug, Args)]
pub struct ColorArgs {
    /// Matrix coefficients
    #[arg(long, value_enum, default_value_t = MatrixChoice::Auto)]
    pub matrix: MatrixChoice,

    /// Value range of the YUV planes
    #[arg(long, value_enum, default_value_t = RangeChoice::Auto)]
    pub range: RangeChoice,
}

impl ColorArgs {
    /// The conversion parameters: the file's signalling, with any override applied.
    pub fn resolve(&self, stream: &SequenceHeader) -> ColorParams {
        let mut params = ColorParams::from_stream(stream);
        match self.matrix {
            MatrixChoice::Auto => {}
            MatrixChoice::Bt601 => params.matrix = ColorMatrix::Bt601,
            MatrixChoice::Bt709 => params.matrix = ColorMatrix::Bt709,
            MatrixChoice::Bt2020 => params.matrix = ColorMatrix::Bt2020,
        }
        match self.range {
            RangeChoice::Auto => {}
            RangeChoice::Full => params.range = YcbcrRange::Full,
            RangeChoice::Limited => params.range = YcbcrRange::Limited,
        }
        params
    }
}

/// A short name for a chroma format.
pub fn chroma_name(chroma: emberwave_core::bitstream::ChromaSubsampling) -> &'static str {
    match chroma {
        emberwave_core::bitstream::ChromaSubsampling::Yuv420 => "4:2:0",
        emberwave_core::bitstream::ChromaSubsampling::Yuv444 => "4:4:4",
    }
}

/// A short name for a chroma siting.
pub fn siting_name(siting: ChromaSiting) -> &'static str {
    match siting {
        ChromaSiting::Center => "centre",
        ChromaSiting::Left => "left",
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Decode(args) => decode::run(&args),
        Command::Encode(args) => encode::run(&args),
        Command::Bench(args) => bench::run(&args),
        Command::BenchEncode(args) => bench_encode::run(&args),
        Command::Info(args) => info::run(&args),
        Command::Compare(args) => compare::run(&args),
        Command::Quality(args) => quality::run(&args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        // Whoever read the output, such as `head`, has all it wanted.
        Err(error) if is_broken_pipe(&*error) => ExitCode::SUCCESS,
        Err(error) => {
            err!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Whether `error` is a write to a pipe whose reader has gone.
fn is_broken_pipe(error: &(dyn std::error::Error + 'static)) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::BrokenPipe)
}
