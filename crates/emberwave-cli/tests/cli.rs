//! End-to-end tests: run the real `emberwave-cli` binary on the committed
//! vectors.
//!
//! The tests that decode need a GPU or a software Vulkan driver. Without one
//! they skip, unless `EMBERWAVE_REQUIRE_GPU` is set (CI sets it on Linux).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use emberwave_core::bitstream::ChromaSubsampling;
use emberwave_core::color::{ColorMatrix, ColorParams, yuv_to_rgb8};

const VECTOR: &str = "gradient_200x120_420";

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_emberwave-cli"))
}

fn testdata(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(file)
}

/// A fresh directory for one test's output.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn gpu_required() -> bool {
    std::env::var_os("EMBERWAVE_REQUIRE_GPU").is_some_and(|value| !value.is_empty())
}

/// Runs a command that needs a GPU. Returns `None`, after saying so, when
/// there is no adapter and none is required.
fn run_on_gpu(command: &mut Command) -> Option<Output> {
    let output = command.output().expect("the binary runs");
    if !output.status.success() && stderr(&output).contains("no suitable GPU adapter") {
        assert!(
            !gpu_required(),
            "a GPU is required but none is available:\n{}",
            stderr(&output)
        );
        eprintln!("no GPU available, skipping");
        return None;
    }
    Some(output)
}

#[test]
fn info_describes_a_vector() {
    let output = cli()
        .arg("info")
        .arg(testdata(&format!("{VECTOR}.pwv1")))
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));

    let text = stdout(&output);
    for expected in [
        "200x120, 4:2:0",
        "padded to 224x128",
        "BT.709 matrix, limited range, centre chroma siting",
        "30/1 (30.000 fps)",
        "frames:        3",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn info_verbose_lists_every_frame() {
    let output = cli()
        .args(["info", "--verbose"])
        .arg(testdata(&format!("{VECTOR}.pwv1")))
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = stdout(&output);
    assert_eq!(text.matches("blocks announced").count(), 3, "{text}");
    // Sequence numbers count up from 1.
    assert!(text.contains("sequence 1"), "{text}");
    assert!(text.contains("sequence 3"), "{text}");
}

#[test]
fn info_rejects_a_file_that_is_not_pwv1() {
    let dir = scratch("info_garbage");
    let path = dir.join("garbage.pwv1");
    std::fs::write(&path, vec![0x55u8; 100]).unwrap();

    let output = cli().arg("info").arg(&path).output().unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("PWV1"), "{}", stderr(&output));

    // A missing file is an error too, not a panic.
    let output = cli()
        .arg("info")
        .arg(dir.join("missing.pwv1"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).starts_with("error:"), "{}", stderr(&output));
}

#[test]
fn compare_judges_planes_by_psnr() {
    let reference = testdata(&format!("{VECTOR}.ref.y4m"));
    let fp32 = testdata(&format!("{VECTOR}.ref-fp32.y4m"));

    // A file against itself is identical.
    let output = cli()
        .arg("compare")
        .args([&reference, &reference])
        .args(["--min-psnr", "100"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stdout(&output).contains("worst PSNR:    inf dB"),
        "{}",
        stdout(&output)
    );

    // The two precisions differ by a rounding or two: fine at 60 dB, not at 100.
    let output = cli()
        .arg("compare")
        .args([&reference, &fp32])
        .args(["--min-psnr", "60"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let output = cli()
        .arg("compare")
        .args([&reference, &fp32])
        .args(["--min-psnr", "100"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("below the required 100"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn compare_refuses_files_of_different_formats() {
    let output = cli()
        .arg("compare")
        .arg(testdata(&format!("{VECTOR}.ref.y4m")))
        .arg(testdata("noise_131x67_444.ref.y4m"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("differ in format"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn compare_does_not_pass_when_there_is_nothing_to_compare() {
    let dir = scratch("compare_nothing");

    // Two files without frames: a "worst PSNR" of infinity would pass any bar.
    let empty = dir.join("empty.y4m");
    std::fs::write(&empty, "YUV4MPEG2 W4 H2 F30:1 C420\n").unwrap();
    let output = cli()
        .arg("compare")
        .args([&empty, &empty])
        .args(["--min-psnr", "100"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("no frames"), "{}", stderr(&output));

    // A header whose picture size overflows must not read as a picture of no
    // samples, which would compare equal to anything.
    let huge = dir.join("huge.y4m");
    std::fs::write(&huge, "YUV4MPEG2 W4294967296 H4294967296 C444\nFRAME\n").unwrap();
    let output = cli().arg("compare").args([&huge, &huge]).output().unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("too large"), "{}", stderr(&output));
}

#[test]
fn options_that_cannot_work_are_refused_before_anything_is_written() {
    let vector = testdata(&format!("{VECTOR}.pwv1"));
    let dir = scratch("refused_options");
    let out_dir = dir.join("never_created");

    let refused = |extra: &[&str], expected: &str| {
        let output = cli()
            .arg("decode")
            .arg(&vector)
            .arg("--out-dir")
            .arg(&out_dir)
            .args(extra)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{extra:?} was accepted");
        assert!(
            stderr(&output).contains(expected),
            "{extra:?}: {}",
            stderr(&output)
        );
        assert!(!out_dir.exists(), "{extra:?} left output behind");
    };
    // Nothing to decode.
    refused(&["--frames", "0"], "--frames must be at least 1");
    refused(
        &["--format", "y4m", "--frames", "0"],
        "--frames must be at least 1",
    );
    // The conversion options mean nothing for planes that are written as they are.
    refused(
        &["--format", "y4m", "--range", "full"],
        "--matrix and --range",
    );
    refused(
        &["--format", "y4m", "--matrix", "bt601"],
        "--matrix and --range",
    );

    let output = cli()
        .arg("bench")
        .arg(&vector)
        .args(["--frames", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--frames must be at least 1"),
        "{}",
        stderr(&output)
    );
    for bad in ["nan", "-1", "inf"] {
        let output = cli()
            .arg("bench")
            .arg(&vector)
            .args(["--warmup-seconds", bad])
            .output()
            .unwrap();
        assert!(!output.status.success(), "--warmup-seconds {bad}");
    }
}

#[test]
fn decode_to_y4m_matches_the_reference_decoder() {
    let dir = scratch("decode_y4m");
    let vector = testdata(&format!("{VECTOR}.pwv1"));
    let Some(output) = run_on_gpu(
        cli()
            .arg("decode")
            .arg(&vector)
            .arg("--out-dir")
            .arg(&dir)
            .args(["--format", "y4m"]),
    ) else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stdout(&output).contains("decoded 3 frames"),
        "{}",
        stdout(&output)
    );

    let decoded = dir.join(format!("{VECTOR}.y4m"));

    // The file says what the stream says: this vector is limited range with
    // centred chroma.
    let bytes = std::fs::read(&decoded).unwrap();
    let header_end = bytes.iter().position(|&b| b == b'\n').unwrap();
    let header = String::from_utf8_lossy(&bytes[..header_end]).into_owned();
    assert!(
        header.contains("XCOLORRANGE=LIMITED") && header.ends_with(" C420jpeg"),
        "{header}"
    );

    for (reference, min) in [("ref.y4m", "60"), ("ref-fp32.y4m", "80")] {
        let output = cli()
            .arg("compare")
            .arg(&decoded)
            .arg(testdata(&format!("{VECTOR}.{reference}")))
            .args(["--min-psnr", min])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "against {reference}: {}",
            stderr(&output)
        );
    }
}

/// Reads the first frame of an 8-bit 4:2:0 Y4M file: its size and planes.
fn first_frame(path: &Path) -> (usize, usize, [Vec<u8>; 3]) {
    let data = std::fs::read(path).unwrap();
    let newline = data.iter().position(|&b| b == b'\n').unwrap();
    let header = std::str::from_utf8(&data[..newline]).unwrap();
    let field = |tag: char| -> usize {
        header
            .split_whitespace()
            .find_map(|t| t.strip_prefix(tag))
            .unwrap()
            .parse()
            .unwrap()
    };
    let (width, height) = (field('W'), field('H'));
    let (chroma_width, chroma_height) = (width / 2, height / 2);

    let start = newline + 1 + "FRAME\n".len();
    let sizes = [
        width * height,
        chroma_width * chroma_height,
        chroma_width * chroma_height,
    ];
    let mut planes: [Vec<u8>; 3] = Default::default();
    let mut at = start;
    for (plane, size) in planes.iter_mut().zip(sizes) {
        *plane = data[at..at + size].to_vec();
        at += size;
    }
    (width, height, planes)
}

#[test]
fn decode_to_png_converts_the_decoded_planes_correctly() {
    let dir = scratch("decode_png");
    let vector = testdata(&format!("{VECTOR}.pwv1"));
    let decode = |format: &str| {
        run_on_gpu(
            cli()
                .arg("decode")
                .arg(&vector)
                .arg("--out-dir")
                .arg(&dir)
                .args(["--format", format]),
        )
    };
    let Some(output) = decode("png") else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    let output = decode("y4m").unwrap();
    assert!(output.status.success(), "{}", stderr(&output));

    // One PNG per frame, named after the input.
    let png_path = dir.join(format!("{VECTOR}_000000.png"));
    assert!(dir.join(format!("{VECTOR}_000002.png")).exists());
    assert!(!dir.join(format!("{VECTOR}_000003.png")).exists());

    let mut reader = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(&png_path).unwrap(),
    ))
    .read_info()
    .unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!((info.width, info.height), (200, 120));
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    pixels.truncate(info.buffer_size());

    // The same conversion done on the CPU from the Y4M the same decoder wrote.
    let (width, height, [y, cb, cr]) = first_frame(&dir.join(format!("{VECTOR}.y4m")));
    let params = ColorParams {
        matrix: ColorMatrix::Bt709,
        ..ColorParams::default()
    };
    let expected = yuv_to_rgb8(
        [&y, &cb, &cr],
        width as u32,
        height as u32,
        ChromaSubsampling::Yuv420,
        &params,
    )
    .expect("the Y4M planes match its size");
    assert_eq!(pixels.len(), expected.len());

    let differing = pixels.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let largest = pixels
        .iter()
        .zip(&expected)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    // The GPU may fuse a multiply and an add, which moves a rounding by one.
    assert!(largest <= 1, "largest difference {largest}");
    assert!(
        differing * 100 <= pixels.len(),
        "{differing} of {} samples differ",
        pixels.len()
    );
}

#[test]
fn bench_reports_per_frame_times_as_text_and_as_json() {
    let vector = testdata(&format!("{VECTOR}.pwv1"));

    let Some(output) = run_on_gpu(cli().arg("bench").arg(&vector).args([
        "--frames",
        "6",
        "--warmup",
        "1",
        "--warmup-seconds",
        "0",
    ])) else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for expected in ["mean", "p50", "p99", "total"] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    // The whole line: "16 measured" would contain the shorter text too.
    assert!(
        text.lines()
            .any(|line| line == "frames:   6 measured after 1 warm-up frames"),
        "{text}"
    );

    let output = run_on_gpu(cli().arg("bench").arg(&vector).args([
        "--frames",
        "4",
        "--warmup",
        "0",
        "--warmup-seconds",
        "0",
        "--json",
        "--output",
        "rgb",
    ]))
    .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(json["frames"], 4);
    assert_eq!(json["warmup"], 0);
    assert_eq!(json["readback"], false);
    assert_eq!(json["unit"], "ms");
    let total = &json["series"]["total"];
    for key in ["mean", "p50", "p99", "min", "max"] {
        assert!(
            total[key].as_f64().is_some_and(|v| v > 0.0),
            "{key}: {total}"
        );
    }
    assert!(total["min"].as_f64() <= total["p50"].as_f64());
    assert!(total["p50"].as_f64() <= total["max"].as_f64());

    // With a read-back there is only the total.
    let output = run_on_gpu(cli().arg("bench").arg(&vector).args([
        "--frames",
        "2",
        "--warmup",
        "0",
        "--warmup-seconds",
        "0",
        "--json",
        "--readback",
    ]))
    .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["readback"], true);
    assert!(
        json["series"]["total"]["mean"]
            .as_f64()
            .is_some_and(|v| v > 0.0)
    );
}

#[test]
fn an_unknown_backend_name_is_rejected_by_the_parser() {
    let output = cli()
        .args(["info", "--backend", "opengl"])
        .arg(testdata(&format!("{VECTOR}.pwv1")))
        .output()
        .unwrap();
    // `info` has no --backend; `decode` does, and only accepts the known names.
    assert!(!output.status.success());

    let output = cli()
        .args(["decode", "--backend", "opengl", "--out-dir", "unused"])
        .arg(testdata(&format!("{VECTOR}.pwv1")))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("possible values"),
        "{}",
        stderr(&output)
    );
}

/// The encoder's vectors: name and bytes per frame (testdata/encoder/vectors.txt).
fn encoder_vectors() -> Vec<(String, String)> {
    std::fs::read_to_string(testdata("encoder/vectors.txt"))
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            (fields[0].to_owned(), fields[6].to_owned())
        })
        .collect()
}

#[test]
fn encode_makes_the_committed_streams_again() {
    // The whole file: the PWV1 header with the stream's size, chroma format,
    // range, siting and frame rate taken from the Y4M header, the size of
    // every frame and the frames themselves.
    let dir = scratch("encode_vectors");
    for (name, budget) in encoder_vectors() {
        let out = dir.join(format!("{name}.pwv1"));
        let output = cli()
            .arg("encode")
            .arg(testdata(&format!("encoder/{name}.source.y4m")))
            .arg("--out")
            .arg(&out)
            .args(["--bytes-per-frame", &budget])
            .output()
            .unwrap();
        assert!(output.status.success(), "{name}: {}", stderr(&output));
        let written = std::fs::read(&out).unwrap();
        let committed = std::fs::read(testdata(&format!("encoder/{name}.pwv1"))).unwrap();
        assert!(written == committed, "{name}: the file differs");
    }
}

#[test]
fn encode_refuses_before_it_writes_and_leaves_no_partial_file() {
    let dir = scratch("encode_refusals");
    let source = testdata("encoder/scene_200x120_420.source.y4m");
    let out = dir.join("out.pwv1");
    std::fs::write(&out, b"what was there before").unwrap();

    let empty = dir.join("empty.y4m");
    std::fs::write(&empty, b"YUV4MPEG2 W4 H2 F30:1 C444\n").unwrap();
    let odd = dir.join("odd.y4m");
    std::fs::write(&odd, b"YUV4MPEG2 W5 H3 C420\nFRAME\n").unwrap();

    for (input, extra, message) in [
        (&source, vec!["--bytes-per-frame", "7"], "8 bytes"),
        (
            &source,
            vec!["--bytes-per-frame", "100", "--frames", "0"],
            "nothing to encode",
        ),
        (&empty, vec!["--bytes-per-frame", "100"], "no frames"),
        (&odd, vec!["--bytes-per-frame", "100"], "even"),
        (
            &source,
            vec!["--bytes-per-frame", "100", "--backend", "metal"],
            "--gpu",
        ),
        (
            &source,
            vec!["--bytes-per-frame", "100", "--software"],
            "--gpu",
        ),
    ] {
        let output = cli()
            .arg("encode")
            .arg(input)
            .arg("--out")
            .arg(&out)
            .args(&extra)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{extra:?}");
        assert!(
            stderr(&output).contains(message),
            "{extra:?}: {}",
            stderr(&output)
        );
        assert_eq!(
            std::fs::read(&out).unwrap(),
            b"what was there before",
            "{extra:?}"
        );
        assert!(!dir.join("out.pwv1.partial").exists(), "{extra:?}");
    }
}

#[test]
fn encode_on_the_gpu_makes_streams_like_the_cpus() {
    // The GPU's frames are the CPU's, or differ in a few blocks (decision
    // D29); the file around them is the same.
    let dir = scratch("encode_gpu");
    let (name, budget) = encoder_vectors()
        .into_iter()
        .find(|(name, _)| name == "scene_200x120_420")
        .unwrap();
    let out = dir.join("gpu.pwv1");
    let Some(output) = run_on_gpu(
        cli()
            .arg("encode")
            .arg(testdata(&format!("encoder/{name}.source.y4m")))
            .arg("--out")
            .arg(&out)
            .args(["--bytes-per-frame", &budget, "--gpu"]),
    ) else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("on the GPU"),
        "{}",
        stderr(&output)
    );
    let written = std::fs::read(&out).unwrap();
    let committed = std::fs::read(testdata(&format!("encoder/{name}.pwv1"))).unwrap();
    // The PWV1 header is the same in any case: 24 bytes.
    assert_eq!(written[..24], committed[..24]);

    // And the stream shows the pictures about as well as the CPU's, whose
    // worst plane is at 41.9 dB.
    let output = run_on_gpu(
        cli()
            .arg("decode")
            .arg(&out)
            .arg("--out-dir")
            .arg(&dir)
            .args(["--format", "y4m"]),
    )
    .expect("a GPU, since the encode found one");
    assert!(output.status.success(), "{}", stderr(&output));
    let output = cli()
        .arg("compare")
        .arg(dir.join("gpu.y4m"))
        .arg(testdata(&format!("encoder/{name}.source.y4m")))
        .args(["--min-psnr", "41"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
}

#[test]
fn bench_encode_reports_per_frame_times_on_the_cpu_and_the_gpu() {
    // Two frames, taken in turn: five measured frames go round them twice.
    let source = testdata("encoder/scene_200x120_420.source.y4m");
    let args = [
        "--bytes-per-frame",
        "6000",
        "--frames",
        "5",
        "--warmup",
        "1",
        "--warmup-seconds",
        "0",
    ];
    let output = cli()
        .arg("bench-encode")
        .arg(&source)
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        text.lines()
            .any(|line| line == "frames:   5 measured after 1 warm-up frames"),
        "{text}"
    );
    assert!(text.contains("encoded on the CPU"), "{text}");

    let Some(output) = run_on_gpu(
        cli()
            .arg("bench-encode")
            .arg(&source)
            .args(args)
            .args(["--gpu", "--json"]),
    ) else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(json["frames"], 5);
    assert_eq!(json["warmup"], 1);
    assert_eq!(json["bytes_per_frame"], 6000);
    assert_eq!(json["picture"], "planes");
    assert_eq!(json["unit"], "ms");
    for series in ["total", "cpu prepare", "gpu wall"] {
        let numbers = &json["series"][series];
        for key in ["mean", "p50", "p99", "min", "max"] {
            assert!(
                numbers[key].as_f64().is_some_and(|v| v > 0.0),
                "{series} {key}: {numbers}"
            );
        }
    }

    // One pass over the video by default.
    let output = cli()
        .arg("bench-encode")
        .arg(&source)
        .args([
            "--bytes-per-frame",
            "6000",
            "--warmup",
            "0",
            "--warmup-seconds",
            "0",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stdout(&output)
            .lines()
            .any(|line| line == "frames:   2 measured after 0 warm-up frames"),
        "{}",
        stdout(&output)
    );
}

#[test]
fn bench_encode_times_pictures_in_textures() {
    let source = testdata("encoder/scene_200x120_420.source.y4m");
    let args = [
        "--bytes-per-frame",
        "6000",
        "--frames",
        "3",
        "--warmup",
        "1",
        "--warmup-seconds",
        "0",
    ];
    // Textures are the GPU's: without --gpu the option is refused.
    let output = cli()
        .arg("bench-encode")
        .arg(&source)
        .args(args)
        .arg("--textures")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--gpu"), "{}", stderr(&output));

    let Some(output) = run_on_gpu(cli().arg("bench-encode").arg(&source).args(args).args([
        "--gpu",
        "--textures",
        "--json",
    ])) else {
        return;
    };
    assert!(output.status.success(), "{}", stderr(&output));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(json["picture"], "textures");
    assert!(
        json["description"]
            .as_str()
            .is_some_and(|text| text.ends_with("encoded on the GPU from textures")),
        "{json}"
    );
    assert!(
        json["series"]["total"]["mean"]
            .as_f64()
            .is_some_and(|mean| mean > 0.0),
        "{json}"
    );
}

#[test]
fn output_into_a_closed_pipe_ends_quietly() {
    // As when the output goes to `head`: the reader is gone before the first
    // line, which is no reason for an error, let alone a panic.
    let (reader, writer) = std::io::pipe().expect("a pipe");
    drop(reader);
    let output = cli()
        .arg("info")
        .arg(testdata(&format!("{VECTOR}.pwv1")))
        .arg("--verbose")
        .stdout(writer)
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the binary runs")
        .wait_with_output()
        .expect("it ends");
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stderr(&output), "");
}

#[test]
fn errors_name_the_file_and_a_psnr_must_be_a_number() {
    let missing = scratch("errors").join("missing.pwv1");
    for command in ["info", "decode"] {
        let mut run = cli();
        run.arg(command).arg(&missing);
        if command == "decode" {
            run.arg("--out-dir").arg(scratch("errors-out"));
        }
        let output = run.output().unwrap();
        assert!(!output.status.success());
        assert!(
            stderr(&output).contains("missing.pwv1"),
            "{command}: {}",
            stderr(&output)
        );
    }
    let reference = testdata(&format!("{VECTOR}.ref.y4m"));
    let output = cli()
        .arg("compare")
        .arg(missing.with_extension("y4m"))
        .arg(&reference)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("missing.y4m"),
        "{}",
        stderr(&output)
    );

    // No PSNR is below NaN, so a NaN minimum would pass anything.
    let output = cli()
        .arg("compare")
        .arg(&reference)
        .arg(&reference)
        .args(["--min-psnr", "nan"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(stderr(&output).contains("finite"), "{}", stderr(&output));
}

#[test]
fn a_file_without_frames_is_an_error_for_every_command() {
    // The 24 bytes of a PWV1 header and no frame.
    let dir = scratch("no-frames");
    let header_only = dir.join("empty.pwv1");
    let bytes = std::fs::read(testdata(&format!("{VECTOR}.pwv1"))).unwrap();
    std::fs::write(&header_only, &bytes[..24]).unwrap();
    for command in ["decode", "bench"] {
        let mut run = cli();
        run.arg(command).arg(&header_only);
        if command == "decode" {
            run.arg("--out-dir").arg(dir.join("out"));
        }
        let Some(output) = run_on_gpu(&mut run) else {
            return;
        };
        assert!(!output.status.success(), "{command}");
        assert!(
            stderr(&output).contains("no frames"),
            "{command}: {}",
            stderr(&output)
        );
    }
}
