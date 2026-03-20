// Title  : CUDA program to count occurrences of a word in a sentence
// Author : Aditya Sinha
// Date   : 13/03/2026

#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__ void countWord(char *sentence, char *word, int slen, int wlen, int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= slen - wlen)
    {
        int match = 1;

        for (int j = 0; j < wlen; j++)
        {
            if (sentence[i + j] != word[j])
            {
                match = 0;
                break;
            }
        }

        if (match)
            atomicAdd(count, 1);
    }
}

int main()
{
    char sentence[200], word[50];
    char *d_sentence, *d_word;
    int *d_count, count = 0;

    printf("Enter sentence: ");
    fgets(sentence, 200, stdin);

    printf("Enter word: ");
    scanf("%s", word);

    int slen = strlen(sentence);
    int wlen = strlen(word);

    cudaMalloc(&d_sentence, slen);
    cudaMalloc(&d_word, wlen);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_sentence, sentence, slen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    countWord<<<1, slen>>>(d_sentence, d_word, slen, wlen, d_count);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Word occurs %d times\n", count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}