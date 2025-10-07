#include <stdio.h>

typedef struct {
    int page;
    int refBit;
} Frame;

int main() {
    int n, framesCount, i, j, pageFaults = 0, hits = 0;
    int next = 0; // pointer for circular replacement

    printf("Enter number of pages: ");
    scanf("%d", &n);

    int pages[n];
    printf("Enter the reference string: ");
    for (i = 0; i < n; i++)
        scanf("%d", &pages[i]);

    printf("Enter number of frames: ");
    scanf("%d", &framesCount);

    Frame frames[framesCount];
    for (i = 0; i < framesCount; i++) {
        frames[i].page = -1;
        frames[i].refBit = 0;
    }

    for (i = 0; i < n; i++) {
        int found = 0;

        // Check if page is already in frame
        for (j = 0; j < framesCount; j++) {
            if (frames[j].page == pages[i]) {
                found = 1;
                hits++;
                frames[j].refBit = 1; // give second chance
                break;
            }
        }

        if (!found) {
            // Page fault
            pageFaults++;

            // Find a frame to replace using second chance
            while (1) {
                if (frames[next].refBit == 0) {
                    frames[next].page = pages[i];
                    frames[next].refBit = 1;
                    next = (next + 1) % framesCount;
                    break;
                } else {
                    frames[next].refBit = 0; // give second chance
                    next = (next + 1) % framesCount;
                }
            }
        }

        // Display current frame contents
        printf("\nAfter reference %d: ", pages[i]);
        for (j = 0; j < framesCount; j++) {
            if (frames[j].page != -1)
                printf("%d(%d) ", frames[j].page, frames[j].refBit);
            else
                printf("- ");
        }
    }

    printf("\n\nTotal Page Faults = %d", pageFaults);
    printf("\nTotal Hits = %d", hits);
    printf("\nHit Ratio = %.2f\n", (float)hits / n);

    return 0;
}
