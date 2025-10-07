#include <stdio.h>

typedef struct {
    int base;
    int limit;
} Segment;

Segment segmentTable[5];

void createSegmentTable() {
    segmentTable[0].base = 4000; segmentTable[0].limit = 1000;

    segmentTable[1].base = 1000; segmentTable[1].limit = 300;

    segmentTable[2].base = 2000; segmentTable[2].limit = 900;

    segmentTable[3].base = 2500; segmentTable[3].limit = 700;

    segmentTable[4].base = 3500; segmentTable[4].limit = 400;
}

int logicalToPhysical(int segment, int offset) {
    if (segment < 0 || segment >= 5) {
        printf("Invalid segment number %d\n", segment);
        return -1;
    }

    if (offset < 0 || offset >= segmentTable[segment].limit) {
        printf("Offset %d out of range for segment %d\n", offset, segment);
        return -1;
    }

    return segmentTable[segment].base + offset;
}

int main() {
    createSegmentTable();

    printf("Physical address for 53 byte of segment 2: %d\n", logicalToPhysical(2, 53));
    printf("Physical address for 852 byte of segment 3: %d\n", logicalToPhysical(3, 852));
    printf("Physical address for 1222 byte of segment 0: %d\n", logicalToPhysical(0, 1222));

    return 0;
}
