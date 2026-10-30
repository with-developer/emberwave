/*
 * Decodes a PWV1 file with the emberwave C API and writes the frames to a Y4M
 * file: the smallest complete use of the API.
 *
 *     emberwave_decode INPUT.pwv1 OUTPUT.y4m
 *
 * A PWV1 file is a 24 byte header followed by frames, each a little-endian
 * 32-bit size and that many bytes of packets. A network client gets the same
 * packets from its transport, and pushes each one as it arrives.
 */
#include <stdio.h>
#include <stdlib.h>

#include "emberwave.h"

static int fail(const char *what, EmberwaveStatus status)
{
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status, emberwave_last_error());
    return 1;
}

/* Reads a whole file into memory. Returns NULL, having said why, on failure. */
static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *data = NULL;
    size_t capacity = 0;
    int error;

    if (!file) {
        perror(path);
        return NULL;
    }
    *size = 0;
    for (;;) {
        size_t got;
        if (*size == capacity) {
            unsigned char *grown;
            capacity = capacity ? 2 * capacity : (size_t)1 << 20;
            grown = realloc(data, capacity);
            if (!grown) {
                fprintf(stderr, "%s: out of memory\n", path);
                free(data);
                fclose(file);
                return NULL;
            }
            data = grown;
        }
        got = fread(data + *size, 1, capacity - *size, file);
        if (got == 0) {
            break;
        }
        *size += got;
    }
    error = ferror(file);
    fclose(file);
    if (error) {
        perror(path);
        free(data);
        return NULL;
    }
    return data;
}

static size_t read_le32(const unsigned char *bytes)
{
    return (size_t)bytes[0] | ((size_t)bytes[1] << 8) | ((size_t)bytes[2] << 16) |
           ((size_t)bytes[3] << 24);
}

int main(int argc, char **argv)
{
    size_t size, offset;
    unsigned char *file, *planes;
    FILE *out;
    EmberwaveStatus status;
    /* A struct passed by pointer starts with its size, which the caller sets. */
    EmberwaveStreamInfo stream = {.struct_size = sizeof stream};
    EmberwaveContext *context = NULL;
    EmberwaveDecoder *decoder = NULL;
    EmberwaveFrameInfo frame = {.struct_size = sizeof frame};
    EmberwavePlane y, cb, cr;
    size_t luma_bytes, chroma_bytes;
    unsigned frames = 0;
    const char *chroma_tag;

    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT.pwv1 OUTPUT.y4m\n", argv[0]);
        return 2;
    }
    file = read_file(argv[1], &size);
    if (!file) {
        return 1;
    }

    /* The file header says how large the pictures are. */
    status = emberwave_probe_pwv1(file, size, &stream);
    if (status != EMBERWAVE_STATUS_OK) {
        return fail("emberwave_probe_pwv1", status);
    }

    /* One context per GPU; its decoders share the compiled shaders. */
    status = emberwave_context_create(NULL, &context);
    if (status != EMBERWAVE_STATUS_OK) {
        return fail("emberwave_context_create", status);
    }
    printf("emberwave %s on %s\n", emberwave_version(), emberwave_context_adapter(context));

    status = emberwave_decoder_create(context, stream.width, stream.height, stream.chroma, &decoder);
    if (status != EMBERWAVE_STATUS_OK) {
        return fail("emberwave_decoder_create", status);
    }
    status = emberwave_decoder_info(decoder, &frame);
    if (status != EMBERWAVE_STATUS_OK) {
        return fail("emberwave_decoder_info", status);
    }

    /* The three planes, back to back, as a Y4M frame holds them. */
    luma_bytes = (size_t)frame.width * frame.height;
    chroma_bytes = (size_t)frame.chroma_width * frame.chroma_height;
    planes = malloc(luma_bytes + 2 * chroma_bytes);
    if (!planes) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    y.data = planes;
    y.stride = frame.width;
    y.size = luma_bytes;
    cb.data = planes + luma_bytes;
    cb.stride = frame.chroma_width;
    cb.size = chroma_bytes;
    cr.data = planes + luma_bytes + chroma_bytes;
    cr.stride = frame.chroma_width;
    cr.size = chroma_bytes;

    out = fopen(argv[2], "wb");
    if (!out) {
        perror(argv[2]);
        return 1;
    }
    if (frame.chroma == EMBERWAVE_CHROMA_444) {
        chroma_tag = "C444";
    } else if (stream.chroma_siting == EMBERWAVE_SITING_LEFT) {
        chroma_tag = "C420mpeg2";
    } else {
        chroma_tag = "C420jpeg";
    }
    fprintf(out, "YUV4MPEG2 W%u H%u F%u:%u Ip A1:1 XCOLORRANGE=%s %s\n", (unsigned)frame.width,
            (unsigned)frame.height,
            (unsigned)(stream.frame_rate_num && stream.frame_rate_den ? stream.frame_rate_num : 30),
            (unsigned)(stream.frame_rate_num && stream.frame_rate_den ? stream.frame_rate_den : 1),
            stream.range == EMBERWAVE_RANGE_FULL ? "FULL" : "LIMITED", chroma_tag);

    offset = 24; /* past the file header */
    while (offset + 4 <= size) {
        size_t frame_size = read_le32(file + offset);
        offset += 4;
        if (frame_size > size - offset) {
            fprintf(stderr, "frame %u is cut off\n", frames);
            return 1;
        }

        /* A whole frame of the file in one call; a network client would
         * push each packet as it arrives. */
        status = emberwave_decoder_push_packet(decoder, file + offset, frame_size);
        offset += frame_size;
        if (status != EMBERWAVE_STATUS_OK) {
            return fail("emberwave_decoder_push_packet", status);
        }
        if (!emberwave_decoder_is_ready(decoder, false)) {
            fprintf(stderr, "frame %u is incomplete\n", frames);
            return 1;
        }

        status = emberwave_decoder_decode_yuv(decoder, y, cb, cr);
        if (status != EMBERWAVE_STATUS_OK) {
            return fail("emberwave_decoder_decode_yuv", status);
        }
        fputs("FRAME\n", out);
        fwrite(planes, 1, luma_bytes + 2 * chroma_bytes, out);
        frames++;
    }
    if (offset != size) {
        fprintf(stderr, "the file ends inside the size of frame %u\n", frames);
        return 1;
    }
    printf("decoded %u frames of %ux%u\n", frames, (unsigned)frame.width, (unsigned)frame.height);

    if (fclose(out) != 0) {
        perror(argv[2]);
        return 1;
    }
    free(planes);
    free(file);
    emberwave_decoder_destroy(decoder);
    emberwave_context_destroy(context);
    return 0;
}
