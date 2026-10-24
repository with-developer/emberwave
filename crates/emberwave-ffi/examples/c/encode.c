/*
 * Encodes the frames of a Y4M video with the emberwave C API, splits each
 * frame into packets the size of a network datagram, decodes the packets
 * again with the C API's decoder, last packet first, and prints the size and
 * the quality of each frame: what a streaming host and its client do between
 * them, without the network.
 *
 *     emberwave_encode INPUT.y4m BYTES_PER_FRAME [MIN_PSNR]
 *
 * The video is 8-bit 4:2:0 or 4:4:4. With MIN_PSNR the program fails when a
 * plane of a frame comes back below that many dB.
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emberwave.h"

/* Bytes of a packet: what fits a UDP datagram on most networks. */
#define PACKET_BYTES 1200

static int fail(const char *what, EmberwaveStatus status)
{
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status, emberwave_last_error());
    return 1;
}

/* A Y4M video being read: 8-bit planes, one frame after the other. */
typedef struct Video {
    FILE *file;
    unsigned width;
    unsigned height;
    uint32_t chroma; /* EMBERWAVE_CHROMA_* */
} Video;

/* Reads the header line of a Y4M file. Returns 0, having said why, if the
 * file is not 8-bit 4:2:0 or 4:4:4 Y4M. */
static int open_video(const char *path, Video *video)
{
    char line[1024];
    char *token;

    video->file = fopen(path, "rb");
    if (!video->file) {
        perror(path);
        return 0;
    }
    if (!fgets(line, sizeof line, video->file) || strncmp(line, "YUV4MPEG2 ", 10) != 0) {
        fprintf(stderr, "%s: not a Y4M file\n", path);
        return 0;
    }
    video->width = 0;
    video->height = 0;
    video->chroma = EMBERWAVE_CHROMA_420;
    for (token = strtok(line + 10, " \n"); token; token = strtok(NULL, " \n")) {
        if (token[0] == 'W') {
            video->width = (unsigned)strtoul(token + 1, NULL, 10);
        } else if (token[0] == 'H') {
            video->height = (unsigned)strtoul(token + 1, NULL, 10);
        } else if (token[0] == 'C') {
            /* 420jpeg, 420mpeg2, 420paldv and 444 are 8-bit; 420p10 and
             * 444p12 are not. */
            int deep = token[4] == 'p' && isdigit((unsigned char)token[5]);
            if (strncmp(token + 1, "444", 3) == 0 && !deep) {
                video->chroma = EMBERWAVE_CHROMA_444;
            } else if (strncmp(token + 1, "420", 3) != 0 || deep) {
                fprintf(stderr, "%s: colour space %s is not 8-bit 4:2:0 or 4:4:4\n", path, token + 1);
                return 0;
            }
        }
    }
    if (video->width == 0 || video->height == 0) {
        fprintf(stderr, "%s: no picture size in the header\n", path);
        return 0;
    }
    return 1;
}

/* Reads the next frame into the planes. Returns 1 for a frame, 0 at the end
 * of the file and -1, having said why, for a broken frame. */
static int read_frame(Video *video, unsigned char *planes[3], const size_t sizes[3])
{
    char magic[5];
    size_t got = fread(magic, 1, sizeof magic, video->file);
    int c;
    int plane;

    if (got == 0) {
        return 0;
    }
    if (got != sizeof magic || memcmp(magic, "FRAME", sizeof magic) != 0) {
        fprintf(stderr, "a frame does not start with FRAME\n");
        return -1;
    }
    /* Frame parameters, however long, up to the end of the line. */
    do {
        c = fgetc(video->file);
        if (c == EOF) {
            fprintf(stderr, "the last frame is cut short\n");
            return -1;
        }
    } while (c != '\n');
    for (plane = 0; plane < 3; plane++) {
        if (fread(planes[plane], 1, sizes[plane], video->file) != sizes[plane]) {
            fprintf(stderr, "the last frame is cut short\n");
            return -1;
        }
    }
    return 1;
}

/* The PSNR of a decoded plane against its source, in dB; 100 for equal
 * planes. */
static double psnr(const unsigned char *source, const unsigned char *decoded, size_t size)
{
    double squared = 0.0;
    size_t i;

    for (i = 0; i < size; i++) {
        double difference = (double)source[i] - (double)decoded[i];
        squared += difference * difference;
    }
    if (squared == 0.0) {
        return 100.0;
    }
    return 10.0 * log10(255.0 * 255.0 * (double)size / squared);
}

int main(int argc, char **argv)
{
    Video video = { NULL, 0, 0, 0 };
    EmberwaveContext *context = NULL;
    EmberwaveEncoder *encoder = NULL;
    EmberwaveDecoder *decoder = NULL;
    EmberwaveFrameInfo info;
    EmberwaveStatus status;
    unsigned char *source[3] = { NULL, NULL, NULL };
    unsigned char *decoded[3] = { NULL, NULL, NULL };
    unsigned char *frame = NULL;
    EmberwavePacket *packets = NULL;
    size_t packet_capacity = 0;
    size_t sizes[3];
    size_t budget;
    double min_psnr = 0.0;
    unsigned frames = 0;
    int result = 1;
    int plane;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s INPUT.y4m BYTES_PER_FRAME [MIN_PSNR]\n", argv[0]);
        return 2;
    }
    budget = (size_t)strtoul(argv[2], NULL, 10);
    if (argc == 4) {
        min_psnr = strtod(argv[3], NULL);
    }
    if (!open_video(argv[1], &video)) {
        goto done;
    }

    /* NULL options: any GPU, a discrete one if there is one. */
    status = emberwave_context_create(NULL, &context);
    if (status != EMBERWAVE_STATUS_OK) {
        result = fail("emberwave_context_create", status);
        goto done;
    }
    printf("GPU: %s\n", emberwave_context_adapter(context));

    /* NULL signalling: BT.709, video range, centred chroma. */
    status = emberwave_encoder_create(context, video.width, video.height, video.chroma, NULL, &encoder);
    if (status != EMBERWAVE_STATUS_OK) {
        result = fail("emberwave_encoder_create", status);
        goto done;
    }
    status = emberwave_decoder_create(context, video.width, video.height, video.chroma, &decoder);
    if (status != EMBERWAVE_STATUS_OK) {
        result = fail("emberwave_decoder_create", status);
        goto done;
    }
    status = emberwave_encoder_info(encoder, &info);
    if (status != EMBERWAVE_STATUS_OK) {
        result = fail("emberwave_encoder_info", status);
        goto done;
    }
    sizes[0] = (size_t)info.width * info.height;
    sizes[1] = sizes[2] = (size_t)info.chroma_width * info.chroma_height;
    for (plane = 0; plane < 3; plane++) {
        source[plane] = malloc(sizes[plane]);
        decoded[plane] = malloc(sizes[plane]);
        if (!source[plane] || !decoded[plane]) {
            fprintf(stderr, "out of memory\n");
            goto done;
        }
    }
    /* A frame is never larger than its budget. */
    frame = malloc(budget ? budget : 1);
    if (!frame) {
        fprintf(stderr, "out of memory\n");
        goto done;
    }

    for (;;) {
        EmberwaveSourcePlane y, cb, cr;
        EmberwavePlane targets[3];
        size_t size, count, i;
        double quality[3];
        int got = read_frame(&video, source, sizes);

        if (got == 0) {
            break;
        }
        if (got < 0) {
            goto done;
        }

        /* Rows back to back here; a capture with padded rows gives its
         * stride instead of the width. */
        y.data = source[0];
        y.stride = info.width;
        cb.data = source[1];
        cb.stride = info.chroma_width;
        cr.data = source[2];
        cr.stride = info.chroma_width;
        status = emberwave_encoder_encode(encoder, y, cb, cr, frame, budget, &size);
        if (status != EMBERWAVE_STATUS_OK) {
            if (status == EMBERWAVE_STATUS_INVALID_ARGUMENT && size > budget) {
                fprintf(stderr, "the smallest frame takes %lu bytes\n", (unsigned long)size);
            }
            result = fail("emberwave_encoder_encode", status);
            goto done;
        }

        /* How many packets, then where they are. */
        status = emberwave_frame_packets(frame, size, PACKET_BYTES, NULL, 0, &count);
        if (status != EMBERWAVE_STATUS_OK) {
            result = fail("emberwave_frame_packets", status);
            goto done;
        }
        if (count > packet_capacity) {
            EmberwavePacket *grown = realloc(packets, count * sizeof *packets);
            if (!grown) {
                fprintf(stderr, "out of memory\n");
                goto done;
            }
            packets = grown;
            packet_capacity = count;
        }
        status = emberwave_frame_packets(frame, size, PACKET_BYTES, packets, packet_capacity, &count);
        if (status != EMBERWAVE_STATUS_OK) {
            result = fail("emberwave_frame_packets", status);
            goto done;
        }

        /* A client gets the packets in whatever order the network brings
         * them; here the last comes first. */
        for (i = count; i-- > 0;) {
            status = emberwave_decoder_push_packet(decoder, frame + packets[i].offset, packets[i].size);
            if (status != EMBERWAVE_STATUS_OK) {
                result = fail("emberwave_decoder_push_packet", status);
                goto done;
            }
        }
        if (!emberwave_decoder_is_ready(decoder, false)) {
            fprintf(stderr, "frame %u: not all blocks arrived\n", frames);
            goto done;
        }
        for (plane = 0; plane < 3; plane++) {
            targets[plane].data = decoded[plane];
            targets[plane].stride = plane == 0 ? info.width : info.chroma_width;
        }
        status = emberwave_decoder_decode_yuv(decoder, targets[0], targets[1], targets[2]);
        if (status != EMBERWAVE_STATUS_OK) {
            result = fail("emberwave_decoder_decode_yuv", status);
            goto done;
        }

        for (plane = 0; plane < 3; plane++) {
            quality[plane] = psnr(source[plane], decoded[plane], sizes[plane]);
        }
        printf("frame %u: %lu bytes in %lu packets, PSNR %.2f / %.2f / %.2f dB\n", frames,
               (unsigned long)size, (unsigned long)count, quality[0], quality[1], quality[2]);
        for (plane = 0; plane < 3; plane++) {
            if (quality[plane] < min_psnr) {
                fprintf(stderr, "frame %u, plane %d: %.2f dB, below %.2f\n", frames, plane,
                        quality[plane], min_psnr);
                goto done;
            }
        }
        frames++;
    }
    printf("%u frames encoded and decoded\n", frames);
    result = frames > 0 ? 0 : 1;

done:
    for (plane = 0; plane < 3; plane++) {
        free(source[plane]);
        free(decoded[plane]);
    }
    free(frame);
    free(packets);
    emberwave_decoder_destroy(decoder);
    emberwave_encoder_destroy(encoder);
    emberwave_context_destroy(context);
    if (video.file) {
        fclose(video.file);
    }
    return result;
}
