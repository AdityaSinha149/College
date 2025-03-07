#include <stdio.h>
#include <stdlib.h>

int max(int a,int b){
    return a>b ? a : b;
}

void delete(int arr[], int n, int i);

void updateHeap(int arr[], int elem, int n) {
    for (int i =n;i>0;i--) {
        if(arr[i]==elem)
            delete(arr, n, i);
    }
}

void delete(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    int sibling=i+1;

    left = left>=n ? -1 : arr[left];
    right = right>=n ? -1 : arr[right];
    sibling = sibling>=n ? -1 : arr[sibling];

    largest=max(max(left,right),sibling);
    if(largest!=-1){
        arr[i]=largest;
        updateHeap(arr,largest,n);
    }
}

void printHeap(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }
    
    int elem;
    printf("Enter elem to be deleted:");
    scanf("%d",&elem);
    updateHeap(arr, elem, n);

    printf("Heapified List: ");
    printHeap(arr, n-1);
    return 0;
}
