#include <stdio.h>
#include <stdlib.h>

#define TOTAL_BLOCKS 100
#define SPARE_BLOCKS 10
#define MAX_FILES 10

// Structure for bad block mapping
struct BadBlock {
    int bad_block;
    int spare_block;
};

// Structure for file
struct File {
    int file_id;
    int start_block;
    int length;
};

// Global data
int disk[TOTAL_BLOCKS];
struct BadBlock bad_blocks[SPARE_BLOCKS];
int bad_block_count = 0;
int next_spare_block = TOTAL_BLOCKS - SPARE_BLOCKS;

// Function to mark bad blocks
void mark_bad_block(int block) {
    if (bad_block_count >= SPARE_BLOCKS) {
        printf("No spare blocks available!\n");
        return;
    }
    bad_blocks[bad_block_count].bad_block = block;
    bad_blocks[bad_block_count].spare_block = next_spare_block++;
    bad_block_count++;
    printf("Bad block %d mapped to spare block %d\n",
           bad_blocks[bad_block_count - 1].bad_block,
           bad_blocks[bad_block_count - 1].spare_block);
}

// Check if a block is bad and return spare mapping if exists
int check_bad_block(int block) {
    for (int i = 0; i < bad_block_count; i++) {
        if (bad_blocks[i].bad_block == block)
            return bad_blocks[i].spare_block;
    }
    return block; // Normal block
}

// Create file with contiguous allocation
void create_file(struct File *file, int start, int length) {
    file->start_block = start;
    file->length = length;

    for (int i = 0; i < length; i++) {
        int actual_block = check_bad_block(start + i);
        disk[actual_block] = file->file_id;
    }
    printf("File %d allocated from block %d to %d\n", file->file_id, start, start + length - 1);
}

// Read file
void read_file(struct File file) {
    printf("Reading File %d:\n", file.file_id);
    for (int i = 0; i < file.length; i++) {
        int actual_block = check_bad_block(file.start_block + i);
        printf("  Block %d -> Actual Block %d -> Data: %d\n",
               file.start_block + i, actual_block, disk[actual_block]);
    }
}

// Show bad block table
void show_bad_block_table() {
    printf("\nBad Blocks Table:\n");
    printf("Bad Block | Spare Block\n");
    for (int i = 0; i < bad_block_count; i++) {
        printf("    %d     |     %d\n", bad_blocks[i].bad_block, bad_blocks[i].spare_block);
    }
}

int main() {
    struct File files[MAX_FILES];

    // Mark a few bad blocks
    mark_bad_block(12);
    // Create some files
    files[0].file_id = 1;
    create_file(&files[0], 10, 5); // includes block 12 (bad)

    // Show bad block table
    show_bad_block_table();

    // Read the files
    printf("\n");
    read_file(files[0]);
    read_file(files[1]);

    return 0;
}
