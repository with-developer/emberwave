// refgen: test vector generator and reference decoder driver for emberwave.
//
// Links against the upstream PyroWave Metal library (see testdata/README.md).
// Encoding and decoding are done entirely by upstream; this tool only
// synthesises input pictures, writes the PWV1 container and reads the decoded
// planes back. It is test tooling and not part of the emberwave library.
//
//   refgen encode --out F.pwv1 --pattern solid|gradient|noise --width W
//                 --height H --chroma 420|444 --frames N --bytes-per-frame B
//                 [--seed S] [--range full|limited] [--siting center|left]
//   refgen decode IN.pwv1 OUT.y4m [--bit-depth 8|16]
//
// PWV1 container layout: bitstream/bitstream.md in the upstream repository,
// section "Raw disk format specification".

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
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

struct Options
{
	std::string out;
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
		else if (key == "--pattern")
			o.pattern = value();
		else if (key == "--width")
			o.width = atoi(value());
		else if (key == "--height")
			o.height = atoi(value());
		else if (key == "--chroma")
			o.chroma = atoi(value());
		else if (key == "--frames")
			o.frames = atoi(value());
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

	if (o.out.empty() || o.width <= 0 || o.height <= 0 || o.frames <= 0 || o.bytes_per_frame == 0)
		fail("encode needs --out, --width, --height, --frames and --bytes-per-frame");
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
	stream.frame_rate_num = FrameRateNum;
	stream.frame_rate_den = FrameRateDen;
	stream.bit_depth = 8;

	File out = open_file(o.out, "wb");
	write_pwv1_header(out.get(), stream);

	// The packer may overshoot the target a little, so leave generous slack.
	std::vector<uint8_t> bitstream(2 * o.bytes_per_frame + (1u << 20));

	for (int frame = 0; frame < o.frames; frame++)
	{
		Picture pic = synthesize(o.pattern, frame, o.width, o.height, chroma_444, o.seed);

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
} // namespace

int main(int argc, char **argv)
{
	if (argc >= 2 && std::string(argv[1]) == "encode")
		return run_encode(argc, argv);
	if (argc >= 2 && std::string(argv[1]) == "decode")
		return run_decode(argc, argv);
	fprintf(stderr, "usage: refgen encode|decode ...\n");
	return 2;
}
