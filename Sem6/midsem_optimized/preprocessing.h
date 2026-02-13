#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reads a source file, removes PHP tags (<?php and ?>), and returns a malloc'ed buffer.
// Caller must free the returned buffer.
static char *preprocessSourceFile(const char *filePath);

// Removes PHP tags from a given buffer and returns a malloc'ed buffer.
// Caller must free the returned buffer.
static char *stripPhpTags(const char *input);

static char *readFile(const char *filePath) {
	FILE *fp = fopen(filePath, "rb");
	if (!fp) {
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	long len = ftell(fp);
	if (len < 0) {
		fclose(fp);
		return NULL;
	}
	rewind(fp);
	char *buf = (char *)malloc((size_t)len + 1);
	if (!buf) {
		fclose(fp);
		return NULL;
	}
	size_t readCount = fread(buf, 1, (size_t)len, fp);
	buf[readCount] = '\0';
	fclose(fp);
	return buf;
}

static char *stripPhpTags(const char *input) {
	if (!input) {
		return NULL;
	}
	size_t len = strlen(input);
	char *out = (char *)malloc(len + 1);
	if (!out) {
		return NULL;
	}

	size_t i = 0;
	size_t o = 0;
	while (i < len) {
		if (i + 5 <= len && strncmp(&input[i], "<?php", 5) == 0) {
			i += 5;
			continue;
		}
		if (i + 2 <= len && strncmp(&input[i], "?>", 2) == 0) {
			i += 2;
			continue;
		}
		out[o++] = input[i++];
	}
	out[o] = '\0';
	return out;
}

static char *preprocessSourceFile(const char *filePath) {
	char *raw = readFile(filePath);
	if (!raw) {
		return NULL;
	}
	char *stripped = stripPhpTags(raw);
	free(raw);
	return stripped;
}

#endif
