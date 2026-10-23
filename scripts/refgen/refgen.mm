// refgen: test vector generator and reference decoder driver for emberwave.
//
// Links against the upstream PyroWave Metal library (see testdata/README.md).
// Encoding and decoding are done entirely by upstream; this tool only
// synthesises input pictures, writes the PWV1 container and reads the decoded
// planes back. It is test tooling and not part of the emberwave library.
//
//   refgen encode --out F.pwv1 --pattern solid|gradient|noise|scene --width W
//                 --height H --chroma 420|444 --frames N --bytes-per-frame B
//                 [--seed S] [--range full|limited] [--siting center|left]
//                 [--source S.y4m]
//   refgen encode --out F.pwv1 --input IN.y4m --bytes-per-frame B [--frames N]
//
// --source also writes the synthesised pictures, as the encoder got them, to
// an 8-bit Y4M file. --input encodes the frames of an 8-bit 4:2:0 or 4:4:4
// Y4M file instead (all of them, or the first N); the picture size, chroma
// format, frame rate and range come from its header. Together they let
// emberwave's encoder and upstream's encode the same pictures.
//   refgen decode IN.pwv1 OUT.y4m [--bit-depth 8|16]
//   refgen bench IN.pwv1 [--frames N] [--warmup N] [--warmup-seconds S] [--json]
//   refgen bench-encode IN.y4m --bytes-per-frame B [--frames N] [--warmup N]
//                 [--warmup-seconds S] [--json]
//
// bench times upstream's decoder the way `emberwave-cli bench --output yuv`
// times emberwave's: planes left in GPU memory, the same warm-up rule, the
// same statistics and the same output, so the two can be put side by side.
// bench-encode times upstream's encoder the way `emberwave-cli bench-encode`
// times emberwave's: from the planes in CPU memory to the frame's packet in
// CPU memory, the video's frames in turn.
//
// PWV1 container layout: bitstream/bitstream.md in the upstream repository,
// section "Raw disk format specification".

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pyrowave_metal.h"

namespace
{
// PWV1Header::Magic: "PWV1" packed with the first letter in the low bits.
constexpr uint32_t Pwv1Magic = 'P' | ('W' << 8) | ('V' << 16) | ('1' << 24);
constexpr size_t Pwv1HeaderSize = 24;
constexpr uint8_t Pwv1HeaderVersion = 1;
constexpr uint32_t FrameRateNum = 30;
constexpr uint32_t FrameRateDen = 1;

// Large enough that a whole frame always lands in a single packet, which is
// what the upstream command line encoder writes per frame.
constexpr size_t SinglePacketBoundary = size_t(1) << 30;

[[noreturn]] void fail(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "refgen: ");
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);
	exit(1);
}

void check(pyrowave_result result, const char *what)
{
	if (result != PYROWAVE_SUCCESS)
		fail("%s failed: %s", what, pyrowave_result_to_string(result));
}

struct FileCloser
{
	void operator()(FILE *f) const
	{
		if (f)
			fclose(f);
	}
};
using File = std::unique_ptr<FILE, FileCloser>;

File open_file(const std::string &path, const char *mode)
{
	File f(fopen(path.c_str(), mode));
	if (!f)
		fail("cannot open %s", path.c_str());
	return f;
}

void put_u32(uint8_t *dst, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		dst[i] = uint8_t(v >> (8 * i));
}

uint32_t get_u32(const uint8_t *src)
{
	return uint32_t(src[0]) | (uint32_t(src[1]) << 8) | (uint32_t(src[2]) << 16) | (uint32_t(src[3]) << 24);
}

struct Stream
{
	int width = 0;
	int height = 0;
	int chroma_444 = 0; // chroma_resolution: 0 = 4:2:0, 1 = 4:4:4
	int siting_left = 0;
	int full_range = 0;
	uint32_t frame_rate_num = 0;
	uint32_t frame_rate_den = 0;
	int bit_depth = 8;
};

// BitstreamSequenceHeader is two little-endian u32 words of bit-fields:
//   word 0: width_minus_1[0..13] height_minus_1[14..27] sequence[28..30] extended[31]
//   word 1: total_blocks[0..23] code[24..25] chroma_resolution[26]
//           color_primaries[27] transfer_function[28] ycbcr_transform[29]
//           ycbcr_range[30] chroma_siting[31]
void write_pwv1_header(FILE *f, const Stream &s)
{
	uint8_t bytes[Pwv1HeaderSize] = {};
	put_u32(bytes + 0, Pwv1Magic);
	put_u32(bytes + 4, uint32_t(s.width - 1) | (uint32_t(s.height - 1) << 14));
	put_u32(bytes + 8, (uint32_t(s.chroma_444) << 26) | (uint32_t(s.full_range ? 0 : 1) << 30) |
	                       (uint32_t(s.siting_left) << 31));
	put_u32(bytes + 12, s.frame_rate_num);
	put_u32(bytes + 16, s.frame_rate_den);
	bytes[20] = uint8_t(s.bit_depth);
	bytes[21] = Pwv1HeaderVersion;
	if (fwrite(bytes, 1, sizeof(bytes), f) != sizeof(bytes))
		fail("short write of PWV1 header");
}

Stream read_pwv1_header(FILE *f)
{
	uint8_t bytes[Pwv1HeaderSize];
	if (fread(bytes, 1, sizeof(bytes), f) != sizeof(bytes))
		fail("short read of PWV1 header");
	if (get_u32(bytes) != Pwv1Magic)
		fail("not a PWV1 file");
	if (bytes[21] != Pwv1HeaderVersion)
		fail("unsupported PWV1 header version %u", bytes[21]);

	const uint32_t w0 = get_u32(bytes + 4);
	const uint32_t w1 = get_u32(bytes + 8);
	Stream s;
	s.width = int(w0 & 0x3fff) + 1;
	s.height = int((w0 >> 14) & 0x3fff) + 1;
	s.chroma_444 = int((w1 >> 26) & 1);
	s.full_range = ((w1 >> 30) & 1) == 0;
	s.siting_left = int((w1 >> 31) & 1);
	s.frame_rate_num = get_u32(bytes + 12);
	s.frame_rate_den = get_u32(bytes + 16);
	s.bit_depth = bytes[20];
	return s;
}

//////
// Picture synthesis. Everything is deterministic so a vector can be reproduced.

// A deterministic "scene" for pictures people can look at: colour bars on top,
// a sky-to-ground gradient, a shaded disc that moves from frame to frame, a
// fine diagonal grid for high-frequency detail, and a little film grain.
struct Rgb
{
	float r, g, b;
};

float hash01(uint32_t x, uint32_t y, uint32_t frame, uint32_t seed)
{
	uint32_t h = x * 374761393u + y * 668265263u + frame * 2246822519u + seed * 3266489917u;
	h = (h ^ (h >> 13)) * 1274126177u;
	h ^= h >> 16;
	return float(h & 0xffffffu) / 16777216.0f;
}

Rgb scene_pixel(int x, int y, int w, int h, int frame, uint32_t seed)
{
	const float u = (float(x) + 0.5f) / float(w);
	const float v = (float(y) + 0.5f) / float(h);
	Rgb c;

	if (v < 0.12f)
	{
		static const float bars[8][3] = {
			{ 1, 1, 1 }, { 1, 1, 0 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 0, 1 }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 0, 0 },
		};
		const int i = std::min(7, int(u * 8.0f));
		c = { bars[i][0], bars[i][1], bars[i][2] };
	}
	else
	{
		const float t = (v - 0.12f) / 0.88f;
		c = { 0.15f + 0.65f * t, 0.35f + 0.25f * t, 0.85f - 0.55f * t };

		// The disc, anti-aliased over about a pixel.
		const float aspect = float(w) / float(h);
		const float cx = 0.30f + 0.40f * (0.5f + 0.5f * sinf(0.15f * float(frame)));
		const float cy = 0.56f;
		const float radius = 0.22f;
		const float dx = (u - cx) * aspect;
		const float dy = v - cy;
		const float coverage = std::clamp(0.5f + (radius - sqrtf(dx * dx + dy * dy)) * float(h), 0.0f, 1.0f);
		const float shade = std::clamp(0.5f + dx / (2.0f * radius), 0.0f, 1.0f);
		const Rgb disc = { 0.95f, 0.80f - 0.50f * shade, 0.20f + 0.60f * shade };
		c = { c.r + (disc.r - c.r) * coverage, c.g + (disc.g - c.g) * coverage, c.b + (disc.b - c.b) * coverage };

		// Fine diagonal grid in the lower left.
		if (u < 0.35f && v > 0.70f && ((x + y) % 16 < 2 || (x - y + 65536) % 16 < 2))
			c = { 0.05f, 0.05f, 0.05f };
	}

	const float grain = (hash01(uint32_t(x), uint32_t(y), uint32_t(frame), seed) - 0.5f) * 0.02f;
	return { std::clamp(c.r + grain, 0.0f, 1.0f), std::clamp(c.g + grain, 0.0f, 1.0f),
		     std::clamp(c.b + grain, 0.0f, 1.0f) };
}

struct Picture
{
	std::vector<uint8_t> plane[3];
	int w[3] = {};
	int h[3] = {};
};

struct Xorshift32
{
	uint32_t state;
	uint32_t next()
	{
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		return state;
	}
};

Picture synthesize(const std::string &pattern, int frame, int width, int height, bool chroma_444, uint32_t seed)
{
	Picture p;
	for (int c = 0; c < 3; c++)
	{
		p.w[c] = (c == 0 || chroma_444) ? width : width / 2;
		p.h[c] = (c == 0 || chroma_444) ? height : height / 2;
		p.plane[c].resize(size_t(p.w[c]) * size_t(p.h[c]));
	}

	if (pattern == "solid")
	{
		std::fill(p.plane[0].begin(), p.plane[0].end(), uint8_t(96 + 16 * frame));
		std::fill(p.plane[1].begin(), p.plane[1].end(), uint8_t(150));
		std::fill(p.plane[2].begin(), p.plane[2].end(), uint8_t(90));
	}
	else if (pattern == "gradient")
	{
		for (int c = 0; c < 3; c++)
		{
			const int dx = std::max(1, p.w[c] - 1);
			const int dy = std::max(1, p.h[c] - 1);
			for (int y = 0; y < p.h[c]; y++)
			{
				for (int x = 0; x < p.w[c]; x++)
				{
					int v;
					if (c == 0)
						v = 16 + (x * 160) / dx + (y * 59) / dy + 6 * frame;
					else if (c == 1)
						v = 64 + (y * 128) / dy;
					else
						v = 192 - (x * 128) / dx;
					p.plane[c][size_t(y) * size_t(p.w[c]) + size_t(x)] = uint8_t(std::min(255, v));
				}
			}
		}
	}
	else if (pattern == "scene")
	{
		// BT.709, video range. Chroma of 4:2:0 is the mean of its 2x2 pixels.
		const int cw = p.w[1], ch = p.h[1];
		const int step = chroma_444 ? 1 : 2;
		for (int cy = 0; cy < ch; cy++)
		{
			for (int cx = 0; cx < cw; cx++)
			{
				float cb_sum = 0.0f, cr_sum = 0.0f;
				for (int dy = 0; dy < step; dy++)
				{
					for (int dx = 0; dx < step; dx++)
					{
						const int x = cx * step + dx, y = cy * step + dy;
						const Rgb c = scene_pixel(x, y, width, height, frame, seed);
						const float luma = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
						cb_sum += (c.b - luma) / 1.8556f;
						cr_sum += (c.r - luma) / 1.5748f;
						p.plane[0][size_t(y) * size_t(width) + size_t(x)] =
						    uint8_t(std::clamp(std::lround(16.0f + 219.0f * luma), 16l, 235l));
					}
				}
				const float n = float(step * step);
				const size_t i = size_t(cy) * size_t(cw) + size_t(cx);
				p.plane[1][i] = uint8_t(std::clamp(std::lround(128.0f + 224.0f * cb_sum / n), 16l, 240l));
				p.plane[2][i] = uint8_t(std::clamp(std::lround(128.0f + 224.0f * cr_sum / n), 16l, 240l));
			}
		}
	}
	else if (pattern == "noise")
	{
		Xorshift32 rng{seed * 2654435761u + uint32_t(frame) * 7919u + 1u};
		for (auto &plane : p.plane)
			for (auto &v : plane)
				v = uint8_t(rng.next() >> 24);
	}
	else
	{
		fail("unknown pattern '%s'", pattern.c_str());
	}

	return p;
}

//////
// encode

// An 8-bit Y4M file read frame by frame.
struct Y4mReader
{
	File file;
	Stream stream;
};

// Opens a Y4M file and reads its header into the stream description: size,
// chroma format, frame rate and range. Only 8-bit 4:2:0 and 4:4:4 will do.
Y4mReader open_y4m(const std::string &path)
{
	Y4mReader r{open_file(path, "rb"), {}};
	char line[512];
	if (!fgets(line, sizeof(line), r.file.get()) || strncmp(line, "YUV4MPEG2 ", 10) != 0)
		fail("%s is not a Y4M file", path.c_str());
	r.stream.frame_rate_num = FrameRateNum;
	r.stream.frame_rate_den = FrameRateDen;
	r.stream.bit_depth = 8;
	std::string chroma = "420";
	char *save = nullptr;
	for (char *token = strtok_r(line + 10, " \n", &save); token; token = strtok_r(nullptr, " \n", &save))
	{
		switch (token[0])
		{
		case 'W':
			r.stream.width = atoi(token + 1);
			break;
		case 'H':
			r.stream.height = atoi(token + 1);
			break;
		case 'C':
			chroma = token + 1;
			break;
		case 'F':
			if (sscanf(token + 1, "%u:%u", &r.stream.frame_rate_num, &r.stream.frame_rate_den) != 2)
				fail("%s: bad frame rate %s", path.c_str(), token);
			break;
		case 'X':
			if (strcmp(token, "XCOLORRANGE=FULL") == 0)
				r.stream.full_range = 1;
			break;
		default:
			break;
		}
	}
	if (chroma == "444")
		r.stream.chroma_444 = 1;
	else if (chroma == "420" || chroma == "420jpeg" || chroma == "420mpeg2")
		r.stream.siting_left = chroma == "420mpeg2";
	else
		fail("%s: only 8-bit 4:2:0 and 4:4:4 Y4M files can be encoded, not C%s", path.c_str(), chroma.c_str());
	if (r.stream.width <= 0 || r.stream.height <= 0)
		fail("%s: the header gives no picture size", path.c_str());
	return r;
}

// Reads the next frame into p, or returns false at the end of the file.
bool read_y4m_frame(Y4mReader &r, Picture &p)
{
	char line[256];
	if (!fgets(line, sizeof(line), r.file.get()))
		return false;
	if (strncmp(line, "FRAME", 5) != 0)
		fail("expected a FRAME line in the Y4M file");
	const Stream &s = r.stream;
	p.w[0] = s.width;
	p.h[0] = s.height;
	p.w[1] = p.w[2] = s.chroma_444 ? s.width : (s.width + 1) / 2;
	p.h[1] = p.h[2] = s.chroma_444 ? s.height : (s.height + 1) / 2;
	for (int c = 0; c < 3; c++)
	{
		p.plane[c].resize(size_t(p.w[c]) * size_t(p.h[c]));
		if (fread(p.plane[c].data(), 1, p.plane[c].size(), r.file.get()) != p.plane[c].size())
			fail("the Y4M file ends inside a frame");
	}
	return true;
}

void write_y4m_header(FILE *f, const Stream &s, int bit_depth);

void write_y4m_frame(FILE *f, const Picture &p)
{
	fputs("FRAME\n", f);
	for (int c = 0; c < 3; c++)
		if (fwrite(p.plane[c].data(), 1, p.plane[c].size(), f) != p.plane[c].size())
			fail("short write of a Y4M frame");
}

struct Options
{
	std::string out;
	std::string input;
	std::string source;
	std::string pattern = "solid";
	int width = 0;
	int height = 0;
	int chroma = 420;
	int frames = 1;
	size_t bytes_per_frame = 0;
	uint32_t seed = 1;
	bool full_range = false;
	bool siting_left = false;
};

int run_encode(int argc, char **argv)
{
	Options o;
	bool frames_given = false;
	for (int i = 2; i < argc; i++)
	{
		const std::string key = argv[i];
		auto value = [&]() -> const char * {
			if (i + 1 >= argc)
				fail("missing value for %s", key.c_str());
			return argv[++i];
		};

		if (key == "--out")
			o.out = value();
		else if (key == "--input")
			o.input = value();
		else if (key == "--source")
			o.source = value();
		else if (key == "--pattern")
			o.pattern = value();
		else if (key == "--width")
			o.width = atoi(value());
		else if (key == "--height")
			o.height = atoi(value());
		else if (key == "--chroma")
			o.chroma = atoi(value());
		else if (key == "--frames")
		{
			o.frames = atoi(value());
			frames_given = true;
		}
		else if (key == "--bytes-per-frame")
			o.bytes_per_frame = size_t(strtoull(value(), nullptr, 0));
		else if (key == "--seed")
			o.seed = uint32_t(strtoul(value(), nullptr, 0));
		else if (key == "--range")
			o.full_range = std::string(value()) == "full";
		else if (key == "--siting")
			o.siting_left = std::string(value()) == "left";
		else
			fail("unknown option %s", key.c_str());
	}

	Y4mReader input;
	if (!o.input.empty())
	{
		input = open_y4m(o.input);
		o.width = input.stream.width;
		o.height = input.stream.height;
		o.chroma = input.stream.chroma_444 ? 444 : 420;
		o.full_range = input.stream.full_range;
		o.siting_left = input.stream.siting_left;
		// All frames unless --frames limits them.
		if (!frames_given)
			o.frames = INT_MAX;
	}
	if (o.out.empty() || o.width <= 0 || o.height <= 0 || o.frames <= 0 || o.bytes_per_frame == 0)
		fail("encode needs --out, --bytes-per-frame, and either --input or --width, --height and --frames");
	if (o.chroma != 420 && o.chroma != 444)
		fail("--chroma must be 420 or 444");
	const bool chroma_444 = o.chroma == 444;
	if (!chroma_444 && ((o.width | o.height) & 1))
		fail("4:2:0 needs even dimensions");

	pyrowave_device device = nullptr;
	check(pyrowave_create_default_device(&device), "create device");

	pyrowave_encoder_create_info info = {};
	info.device = device;
	info.width = o.width;
	info.height = o.height;
	info.chroma = chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_encoder encoder = nullptr;
	check(pyrowave_encoder_create(&info, &encoder), "create encoder");

	Stream stream;
	stream.width = o.width;
	stream.height = o.height;
	stream.chroma_444 = chroma_444;
	stream.siting_left = o.siting_left;
	stream.full_range = o.full_range;
	stream.frame_rate_num = o.input.empty() ? FrameRateNum : input.stream.frame_rate_num;
	stream.frame_rate_den = o.input.empty() ? FrameRateDen : input.stream.frame_rate_den;
	stream.bit_depth = 8;

	File out = open_file(o.out, "wb");
	write_pwv1_header(out.get(), stream);
	File source;
	if (!o.source.empty())
	{
		source = open_file(o.source, "wb");
		write_y4m_header(source.get(), stream, 8);
	}

	// The packer may overshoot the target a little, so leave generous slack.
	std::vector<uint8_t> bitstream(2 * o.bytes_per_frame + (1u << 20));

	for (int frame = 0; frame < o.frames; frame++)
	{
		Picture pic;
		if (o.input.empty())
			pic = synthesize(o.pattern, frame, o.width, o.height, chroma_444, o.seed);
		else if (!read_y4m_frame(input, pic))
			break;
		if (source)
			write_y4m_frame(source.get(), pic);

		pyrowave_cpu_buffer input = {};
		for (int c = 0; c < 3; c++)
		{
			input.data[c] = pic.plane[c].data();
			input.row_stride_in_bytes[c] = size_t(pic.w[c]);
			input.plane_size_in_bytes[c] = pic.plane[c].size();
		}
		input.width = o.width;
		input.height = o.height;
		input.format = chroma_444 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;

		pyrowave_rate_control rate = { o.bytes_per_frame };
		check(pyrowave_encoder_encode_cpu(encoder, &input, &rate), "encode");

		size_t num_packets = 0;
		check(pyrowave_encoder_compute_num_packets(encoder, SinglePacketBoundary, &num_packets), "count packets");
		if (num_packets != 1)
			fail("expected one packet per frame, got %zu", num_packets);

		pyrowave_packet packet = {};
		size_t written = 0;
		check(pyrowave_encoder_packetize(encoder, &packet, SinglePacketBoundary, &written,
		                                 bitstream.data(), bitstream.size()),
		      "packetize");
		if (written != 1)
			fail("packetize produced %zu packets", written);

		uint8_t size_bytes[4];
		put_u32(size_bytes, uint32_t(packet.size));
		if (fwrite(size_bytes, 1, 4, out.get()) != 4 ||
		    fwrite(bitstream.data() + packet.offset, 1, packet.size, out.get()) != packet.size)
			fail("short write");

		fprintf(stderr, "frame %d: %zu bytes (budget %zu)\n", frame, packet.size, o.bytes_per_frame);
	}

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
	return 0;
}

//////
// decode

void write_y4m_header(FILE *f, const Stream &s, int bit_depth)
{
	const char *tag = s.chroma_444 ? (bit_depth == 16 ? "C444p16" : "C444")
	                               : (bit_depth == 16 ? "C420p16" : "C420");
	const uint32_t num = s.frame_rate_num ? s.frame_rate_num : FrameRateNum;
	const uint32_t den = s.frame_rate_den ? s.frame_rate_den : FrameRateDen;
	fprintf(f, "YUV4MPEG2 W%d H%d F%u:%u Ip A1:1 XCOLORRANGE=%s %s\n", s.width, s.height, num, den,
	        s.full_range ? "FULL" : "LIMITED", tag);
}

int run_decode(int argc, char **argv)
{
	if (argc < 4)
		fail("usage: refgen decode IN.pwv1 OUT.y4m [--bit-depth 8|16]");
	const std::string in_path = argv[2];
	const std::string out_path = argv[3];
	int bit_depth_override = 0;
	for (int i = 4; i < argc; i++)
	{
		if (std::string(argv[i]) == "--bit-depth" && i + 1 < argc)
			bit_depth_override = atoi(argv[++i]);
		else
			fail("unknown option %s", argv[i]);
	}

	File in = open_file(in_path, "rb");
	const Stream stream = read_pwv1_header(in.get());
	const int bit_depth = bit_depth_override ? bit_depth_override : (stream.bit_depth == 8 ? 8 : 16);
	if (bit_depth != 8 && bit_depth != 16)
		fail("--bit-depth must be 8 or 16");

	@autoreleasepool
	{
		id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
		if (!mtl_device)
			fail("no Metal device");
		id<MTLCommandQueue> queue = [mtl_device newCommandQueue];

		pyrowave_device_create_info device_info = {};
		device_info.mtl_device = (__bridge void *)mtl_device;
		pyrowave_device device = nullptr;
		check(pyrowave_device_create(&device_info, &device), "create device");

		pyrowave_decoder_create_info info = {};
		info.device = device;
		info.width = stream.width;
		info.height = stream.height;
		info.chroma = stream.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_decoder decoder = nullptr;
		check(pyrowave_decoder_create(&info, &decoder), "create decoder");

		const MTLPixelFormat format = bit_depth == 8 ? MTLPixelFormatR8Unorm : MTLPixelFormatR16Unorm;
		const size_t bytes_per_texel = bit_depth == 8 ? 1 : 2;
		int plane_w[3], plane_h[3];
		id<MTLTexture> textures[3];
		for (int c = 0; c < 3; c++)
		{
			plane_w[c] = (c == 0 || stream.chroma_444) ? stream.width : stream.width / 2;
			plane_h[c] = (c == 0 || stream.chroma_444) ? stream.height : stream.height / 2;
			MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
			                                                                                width:plane_w[c]
			                                                                               height:plane_h[c]
			                                                                            mipmapped:NO];
			desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
			desc.storageMode = MTLStorageModeShared;
			textures[c] = [mtl_device newTextureWithDescriptor:desc];
			if (!textures[c])
				fail("cannot create plane texture %d", c);
		}

		File out = open_file(out_path, "wb");
		write_y4m_header(out.get(), stream, bit_depth);

		std::vector<uint8_t> packet;
		std::vector<uint8_t> plane_bytes;
		int frame = 0;
		for (;; frame++)
		{
			uint8_t size_bytes[4];
			if (fread(size_bytes, 1, 4, in.get()) != 4)
				break;
			packet.resize(get_u32(size_bytes));
			if (fread(packet.data(), 1, packet.size(), in.get()) != packet.size())
				fail("truncated frame %d", frame);

			check(pyrowave_decoder_push_packet(decoder, packet.data(), packet.size()), "push packet");
			if (!pyrowave_decoder_decode_is_ready(decoder, false))
				fail("frame %d is not complete after pushing its packet", frame);

			@autoreleasepool
			{
				id<MTLCommandBuffer> cmd = [queue commandBuffer];
				pyrowave_gpu_buffers buffers = {};
				for (int c = 0; c < 3; c++)
					buffers.planes[c] = (__bridge void *)textures[c];
				check(pyrowave_decoder_decode_gpu_buffer(decoder, (__bridge void *)cmd, &buffers), "decode");
				[cmd commit];
				[cmd waitUntilCompleted];
				if (cmd.error)
					fail("command buffer failed: %s", cmd.error.localizedDescription.UTF8String);
			}

			fputs("FRAME\n", out.get());
			for (int c = 0; c < 3; c++)
			{
				const size_t row_bytes = size_t(plane_w[c]) * bytes_per_texel;
				plane_bytes.resize(row_bytes * size_t(plane_h[c]));
				[textures[c] getBytes:plane_bytes.data()
				          bytesPerRow:row_bytes
				           fromRegion:MTLRegionMake2D(0, 0, plane_w[c], plane_h[c])
				          mipmapLevel:0];
				if (fwrite(plane_bytes.data(), 1, plane_bytes.size(), out.get()) != plane_bytes.size())
					fail("short write");
			}
		}

		fprintf(stderr, "decoded %d frames\n", frame);
		pyrowave_decoder_destroy(decoder);
		pyrowave_device_destroy(device);
	}
	return 0;
}

//////
// bench

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b)
{
	return std::chrono::duration<double, std::milli>(b - a).count();
}

// Nearest-rank percentile of an ascending, non-empty list, as emberwave-cli
// computes it: the smallest sample that at least `percent` of the samples do
// not exceed.
double percentile(const std::vector<double> &sorted, double percent)
{
	const size_t rank = size_t(std::ceil(percent / 100.0 * double(sorted.size())));
	return sorted[std::clamp<size_t>(rank, 1, sorted.size()) - 1];
}

struct Series
{
	const char *name;
	std::vector<double> samples;
};

struct Summary
{
	double mean, p50, p99, min, max;
};

Summary summarize(std::vector<double> samples)
{
	std::sort(samples.begin(), samples.end());
	double sum = 0.0;
	for (double s : samples)
		sum += s;
	return { sum / double(samples.size()), percentile(samples, 50.0), percentile(samples, 99.0),
		     samples.front(), samples.back() };
}

std::string json_string(const std::string &text)
{
	std::string out = "\"";
	for (unsigned char c : text)
	{
		if (c == '"' || c == '\\')
		{
			out += '\\';
			out += char(c);
		}
		else if (c < 0x20)
		{
			char escaped[8];
			snprintf(escaped, sizeof(escaped), "\\u%04x", c);
			out += escaped;
		}
		else
			out += char(c);
	}
	return out + "\"";
}

int run_bench(int argc, char **argv)
{
	if (argc < 3)
		fail("usage: refgen bench IN.pwv1 [--frames N] [--warmup N] [--warmup-seconds S] [--json]");
	const std::string in_path = argv[2];
	long frames_to_measure = 0;
	long warmup_frames = 10;
	double warmup_seconds = 1.0;
	bool json = false;
	for (int i = 3; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "--frames" && i + 1 < argc)
			frames_to_measure = atol(argv[++i]);
		else if (arg == "--warmup" && i + 1 < argc)
			warmup_frames = atol(argv[++i]);
		else if (arg == "--warmup-seconds" && i + 1 < argc)
			warmup_seconds = atof(argv[++i]);
		else if (arg == "--json")
			json = true;
		else
			fail("unknown option %s", argv[i]);
	}
	if (warmup_frames < 0 || !(warmup_seconds >= 0.0 && warmup_seconds <= 3600.0))
		fail("the warm-up must be at least 0 frames and between 0 and 3600 seconds");

	File in = open_file(in_path, "rb");
	const Stream stream = read_pwv1_header(in.get());
	std::vector<std::vector<uint8_t>> frames;
	for (;;)
	{
		uint8_t size_bytes[4];
		if (fread(size_bytes, 1, 4, in.get()) != 4)
			break;
		std::vector<uint8_t> packet(get_u32(size_bytes));
		if (fread(packet.data(), 1, packet.size(), in.get()) != packet.size())
			fail("truncated frame %zu", frames.size());
		frames.push_back(std::move(packet));
	}
	if (frames.empty())
		fail("the file has no frames");
	if (frames_to_measure <= 0)
		frames_to_measure = long(frames.size());

	@autoreleasepool
	{
		id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
		if (!mtl_device)
			fail("no Metal device");
		id<MTLCommandQueue> queue = [mtl_device newCommandQueue];

		pyrowave_device_create_info device_info = {};
		device_info.mtl_device = (__bridge void *)mtl_device;
		pyrowave_device device = nullptr;
		check(pyrowave_device_create(&device_info, &device), "create device");

		pyrowave_decoder_create_info info = {};
		info.device = device;
		info.width = stream.width;
		info.height = stream.height;
		info.chroma = stream.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
		pyrowave_decoder decoder = nullptr;
		check(pyrowave_decoder_create(&info, &decoder), "create decoder");

		// 8-bit planes that only the GPU touches, like emberwave's packed planes.
		pyrowave_gpu_buffers buffers = {};
		id<MTLTexture> textures[3];
		for (int c = 0; c < 3; c++)
		{
			const int w = (c == 0 || stream.chroma_444) ? stream.width : stream.width / 2;
			const int h = (c == 0 || stream.chroma_444) ? stream.height : stream.height / 2;
			MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
			                                                                                width:w
			                                                                               height:h
			                                                                            mipmapped:NO];
			desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
			desc.storageMode = MTLStorageModePrivate;
			textures[c] = [mtl_device newTextureWithDescriptor:desc];
			if (!textures[c])
				fail("cannot create plane texture %d", c);
			buffers.planes[c] = (__bridge void *)textures[c];
		}

		// One frame: the packet goes to the parser (the CPU stage), then the GPU
		// work is encoded, committed and waited for. Clearing first lets the
		// parser take the file's frames again when they repeat, as emberwave's
		// decoder does.
		struct Times
		{
			double prepare, wall, gpu;
		};
		auto decode = [&](size_t index) -> Times {
			const std::vector<uint8_t> &packet = frames[index % frames.size()];
			const Clock::time_point start = Clock::now();
			pyrowave_decoder_clear(decoder);
			check(pyrowave_decoder_push_packet(decoder, packet.data(), packet.size()), "push packet");
			if (!pyrowave_decoder_decode_is_ready(decoder, false))
				fail("frame %zu is not complete after pushing its packet", index % frames.size());
			const Clock::time_point pushed = Clock::now();

			Times times = {};
			@autoreleasepool
			{
				id<MTLCommandBuffer> cmd = [queue commandBuffer];
				check(pyrowave_decoder_decode_gpu_buffer(decoder, (__bridge void *)cmd, &buffers), "decode");
				[cmd commit];
				[cmd waitUntilCompleted];
				const Clock::time_point done = Clock::now();
				if (cmd.error)
					fail("command buffer failed: %s", cmd.error.localizedDescription.UTF8String);
				times = { ms_between(start, pushed), ms_between(pushed, done),
					      (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0 };
			}
			return times;
		};

		// Warm up for a number of frames and for a while, like emberwave-cli.
		size_t warmed = 0;
		const Clock::time_point warmup_start = Clock::now();
		while (warmed < size_t(warmup_frames) || ms_between(warmup_start, Clock::now()) < warmup_seconds * 1000.0)
			decode(warmed++);

		Series total = { "total", {} };
		Series prepare = { "cpu prepare", {} };
		Series wall = { "gpu wall", {} };
		Series gpu = { "gpu (command buffer)", {} };
		for (long i = 0; i < frames_to_measure; i++)
		{
			const Times t = decode(size_t(i));
			total.samples.push_back(t.prepare + t.wall);
			prepare.samples.push_back(t.prepare);
			wall.samples.push_back(t.wall);
			gpu.samples.push_back(t.gpu);
		}

		const char *precision = getenv("PYROWAVE_PRECISION");
		const std::string adapter = mtl_device.name.UTF8String;
		const std::string description = std::to_string(stream.width) + "x" + std::to_string(stream.height) +
		                                (stream.chroma_444 ? " 4:4:4" : " 4:2:0") +
		                                ", decoded by upstream's Metal decoder (PYROWAVE_PRECISION=" +
		                                (precision ? precision : "default") + ") to planes in GPU memory";
		const Series *all[] = { &total, &prepare, &wall, &gpu };

		if (json)
		{
			printf("{\"input\":%s,\"adapter\":%s,\"backend\":\"Metal\",\"description\":%s,\"frames\":%ld,"
			       "\"warmup\":%zu,\"readback\":false,\"unit\":\"ms\",\"series\":{",
			       json_string(in_path).c_str(), json_string(adapter).c_str(), json_string(description).c_str(),
			       frames_to_measure, warmed);
			for (size_t i = 0; i < 4; i++)
			{
				const Summary s = summarize(all[i]->samples);
				printf("%s%s:{\"mean\":%.6f,\"p50\":%.6f,\"p99\":%.6f,\"min\":%.6f,\"max\":%.6f}", i ? "," : "",
				       json_string(all[i]->name).c_str(), s.mean, s.p50, s.p99, s.min, s.max);
			}
			printf("}}\n");
		}
		else
		{
			printf("adapter:  %s (Metal, upstream decoder)\n", adapter.c_str());
			printf("decoding: %s\n", description.c_str());
			printf("frames:   %ld measured after %zu warm-up frames\n\n", frames_to_measure, warmed);
			printf("%-20s%9s%9s%9s%9s%9s   (ms per frame)\n", "", "mean", "p50", "p99", "min", "max");
			for (const Series *series : all)
			{
				const Summary s = summarize(series->samples);
				printf("%-20s%9.3f%9.3f%9.3f%9.3f%9.3f\n", series->name, s.mean, s.p50, s.p99, s.min, s.max);
			}
		}

		pyrowave_decoder_destroy(decoder);
		pyrowave_device_destroy(device);
	}
	return 0;
}
//////
// bench-encode

int run_bench_encode(int argc, char **argv)
{
	if (argc < 3)
		fail("usage: refgen bench-encode IN.y4m --bytes-per-frame B [--frames N] [--warmup N] "
		     "[--warmup-seconds S] [--json]");
	const std::string in_path = argv[2];
	size_t bytes_per_frame = 0;
	long frames_to_measure = 0;
	long warmup_frames = 10;
	double warmup_seconds = 1.0;
	bool json = false;
	for (int i = 3; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "--bytes-per-frame" && i + 1 < argc)
			bytes_per_frame = size_t(atol(argv[++i]));
		else if (arg == "--frames" && i + 1 < argc)
			frames_to_measure = atol(argv[++i]);
		else if (arg == "--warmup" && i + 1 < argc)
			warmup_frames = atol(argv[++i]);
		else if (arg == "--warmup-seconds" && i + 1 < argc)
			warmup_seconds = atof(argv[++i]);
		else if (arg == "--json")
			json = true;
		else
			fail("unknown option %s", argv[i]);
	}
	if (bytes_per_frame == 0)
		fail("bench-encode needs --bytes-per-frame");
	if (warmup_frames < 0 || !(warmup_seconds >= 0.0 && warmup_seconds <= 3600.0))
		fail("the warm-up must be at least 0 frames and between 0 and 3600 seconds");

	// One pass over the video unless --frames says otherwise.
	Y4mReader input = open_y4m(in_path);
	const Stream stream = input.stream;
	if (frames_to_measure <= 0)
	{
		Picture pic;
		while (read_y4m_frame(input, pic))
			frames_to_measure++;
		if (frames_to_measure == 0)
			fail("the Y4M file has no frames");
		input = open_y4m(in_path);
	}

	// The frames in turn, from the start again after the last; reading them
	// is not timed.
	auto next_frame = [&](Picture &pic) {
		if (read_y4m_frame(input, pic))
			return;
		input = open_y4m(in_path);
		if (!read_y4m_frame(input, pic))
			fail("the Y4M file has no frames");
	};

	pyrowave_device device = nullptr;
	check(pyrowave_create_default_device(&device), "create device");
	pyrowave_encoder_create_info info = {};
	info.device = device;
	info.width = stream.width;
	info.height = stream.height;
	info.chroma = stream.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_encoder encoder = nullptr;
	check(pyrowave_encoder_create(&info, &encoder), "create encoder");
	std::vector<uint8_t> bitstream(2 * bytes_per_frame + (1u << 20));

	// One frame: the planes go to the encoder, which submits its GPU work;
	// counting the packets waits for it; packetizing copies the frame out.
	Picture pic;
	auto encode = [&]() -> double {
		next_frame(pic);
		pyrowave_cpu_buffer buffer = {};
		for (int c = 0; c < 3; c++)
		{
			buffer.data[c] = pic.plane[c].data();
			buffer.row_stride_in_bytes[c] = size_t(pic.w[c]);
			buffer.plane_size_in_bytes[c] = pic.plane[c].size();
		}
		buffer.width = stream.width;
		buffer.height = stream.height;
		buffer.format = stream.chroma_444 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
		pyrowave_rate_control rate = { bytes_per_frame };

		const Clock::time_point start = Clock::now();
		check(pyrowave_encoder_encode_cpu(encoder, &buffer, &rate), "encode");
		size_t num_packets = 0;
		check(pyrowave_encoder_compute_num_packets(encoder, SinglePacketBoundary, &num_packets), "count packets");
		if (num_packets != 1)
			fail("expected one packet per frame, got %zu", num_packets);
		pyrowave_packet packet = {};
		size_t written = 0;
		check(pyrowave_encoder_packetize(encoder, &packet, SinglePacketBoundary, &written, bitstream.data(),
		                                 bitstream.size()),
		      "packetize");
		return ms_between(start, Clock::now());
	};

	// Warm up for a number of frames and for a while, like emberwave-cli.
	size_t warmed = 0;
	const Clock::time_point warmup_start = Clock::now();
	while (warmed < size_t(warmup_frames) || ms_between(warmup_start, Clock::now()) < warmup_seconds * 1000.0)
	{
		encode();
		warmed++;
	}

	Series total = { "total", {} };
	for (long i = 0; i < frames_to_measure; i++)
		total.samples.push_back(encode());

	const std::string adapter = [MTLCreateSystemDefaultDevice() name].UTF8String;
	const std::string description = std::to_string(stream.width) + "x" + std::to_string(stream.height) +
	                                (stream.chroma_444 ? " 4:4:4" : " 4:2:0") + ", " +
	                                std::to_string(bytes_per_frame) +
	                                " bytes per frame, encoded by upstream's Metal encoder from planes in CPU memory";
	const Summary s = summarize(total.samples);
	if (json)
	{
		printf("{\"input\":%s,\"adapter\":%s,\"backend\":\"Metal\",\"description\":%s,\"frames\":%ld,"
		       "\"warmup\":%zu,\"bytes_per_frame\":%zu,\"unit\":\"ms\",\"series\":{",
		       json_string(in_path).c_str(), json_string(adapter).c_str(), json_string(description).c_str(),
		       frames_to_measure, warmed, bytes_per_frame);
		printf("%s:{\"mean\":%.6f,\"p50\":%.6f,\"p99\":%.6f,\"min\":%.6f,\"max\":%.6f}", json_string(total.name).c_str(),
		       s.mean, s.p50, s.p99, s.min, s.max);
		printf("}}\n");
	}
	else
	{
		printf("adapter:  %s (Metal, upstream encoder)\n", adapter.c_str());
		printf("encoding: %s\n", description.c_str());
		printf("frames:   %ld measured after %zu warm-up frames\n\n", frames_to_measure, warmed);
		printf("%-20s%9s%9s%9s%9s%9s   (ms per frame)\n", "", "mean", "p50", "p99", "min", "max");
		printf("%-20s%9.3f%9.3f%9.3f%9.3f%9.3f\n", total.name, s.mean, s.p50, s.p99, s.min, s.max);
	}

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
	return 0;
}
} // namespace

int main(int argc, char **argv)
{
	if (argc >= 2 && std::string(argv[1]) == "encode")
		return run_encode(argc, argv);
	if (argc >= 2 && std::string(argv[1]) == "decode")
		return run_decode(argc, argv);
	if (argc >= 2 && std::string(argv[1]) == "bench")
		return run_bench(argc, argv);
	if (argc >= 2 && std::string(argv[1]) == "bench-encode")
		return run_bench_encode(argc, argv);
	fprintf(stderr, "usage: refgen encode|decode|bench|bench-encode ...\n");
	return 2;
}
