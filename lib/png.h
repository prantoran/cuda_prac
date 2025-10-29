
#include <stdio.h>
#include <stdlib.h>
#include <png.h>

// Structure to hold image data
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned char *data; // RGB or RGBA pixel data
} Image;

// Function to read PNG file into Image struct
int read_png_file(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return 0;
    }

    // Read and check PNG signature
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8 || png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "Not a valid PNG file.\n");
        fclose(fp);
        return 0;
    }

    // Create PNG read struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return 0;
    }

    // Create PNG info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return 0;
    }

    // Error handling with setjmp
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return 0;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    // Read PNG info
    png_read_info(png_ptr, info_ptr);
    img->width = png_get_image_width(png_ptr, info_ptr);
    img->height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Convert palette/gray to RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    // Ensure 8-bit depth
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    // Ensure RGBA format
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    // Allocate memory for pixel data
    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    img->data = (unsigned char *)malloc(rowbytes * img->height);
    if (!img->data) {
        fprintf(stderr, "Memory allocation failed.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return 0;
    }

    // Read image row by row
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * img->height);
    for (unsigned int y = 0; y < img->height; y++)
        row_pointers[y] = img->data + y * rowbytes;

    png_read_image(png_ptr, row_pointers);

    // Cleanup
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return 1;
}

// Function to write a grayscale PNG from an unsigned char array
int write_png_grayscale(const char *filename, unsigned char *data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("File opening failed");
        return 1;
    }

    // Create PNG write struct
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        fprintf(stderr, "Failed to create PNG write struct\n");
        return 1;
    }

    // Create PNG info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        fprintf(stderr, "Failed to create PNG info struct\n");
        return 1;
    }

    // Error handling with setjmp
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        fprintf(stderr, "Error during PNG creation\n");
        return 1;
    }

    png_init_io(png_ptr, fp);

    // Set PNG header info (8-bit grayscale)
    png_set_IHDR(
        png_ptr, info_ptr, width, height,
        8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE
    );

    png_write_info(png_ptr, info_ptr);

    // Write image row by row
    png_bytep row_pointers[height];
    for (int y = 0; y < height; y++) {
        row_pointers[y] = data + y * width; // Each row is width bytes
    }
    png_write_image(png_ptr, row_pointers);

    // End write
    png_write_end(png_ptr, NULL);

    // Cleanup
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return 0; // Success
}
