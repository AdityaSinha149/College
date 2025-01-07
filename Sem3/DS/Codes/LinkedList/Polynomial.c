#include <stdio.h>
#include <stdlib.h>

struct Term {
    int coeff;
    int expo;
    struct Term* next;
};

typedef struct Term Node;

// Append a new term to the polynomial
void append(Node** head, int coeff, int expo) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->coeff = coeff;
    newNode->expo = expo;
    newNode->next = NULL;

    if (*head == NULL) {
        *head = newNode;
    } else {
        Node* temp = *head;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}

// Create a polynomial by taking input from the user
void createPoly(Node** head) {
    int n;
    printf("Enter the number of terms: ");
    scanf("%d", &n);

    for (int i = 0; i < n; i++) {
        int coeff, expo;
        printf("Enter coeff and expo of term %d: ", i + 1);
        scanf("%d %d", &coeff, &expo);
        append(head, coeff, expo);
    }
}

// Add two polynomials and return the result
Node* addP(Node* h1, Node* h2) {
    Node* result = NULL;

    while (h1 && h2) {
        if (h1->expo > h2->expo) {
            append(&result, h1->coeff, h1->expo);
            h1 = h1->next;
        } else if (h2->expo > h1->expo) {
            append(&result, h2->coeff, h2->expo);
            h2 = h2->next;
        } else {
            int sum = h1->coeff + h2->coeff;
            if (sum != 0) {
                append(&result, sum, h1->expo);
            }
            h1 = h1->next;
            h2 = h2->next;
        }
    }

    // Append remaining terms
    while (h1) {
        append(&result, h1->coeff, h1->expo);
        h1 = h1->next;
    }
    while (h2) {
        append(&result, h2->coeff, h2->expo);
        h2 = h2->next;
    }

    return result;
}

// Multiply two polynomials and return the result
Node* multiP(Node* h1, Node* h2) {
    Node* result = NULL;

    for (Node* p1 = h1; p1 != NULL; p1 = p1->next) {
        Node* tempResult = NULL;
        for (Node* p2 = h2; p2 != NULL; p2 = p2->next) {
            int coeff = p1->coeff * p2->coeff;
            int expo = p1->expo + p2->expo;
            append(&tempResult, coeff, expo);
        }
        result = addP(result, tempResult);  // Add intermediate result
    }

    return result;
}

// Display the polynomial
void displayPoly(Node* head) {
    Node* p = head;
    while (p != NULL) {
        printf("%dx^%d", p->coeff, p->expo);
        if (p->next != NULL) {
            printf(" + ");
        }
        p = p->next;
    }
    printf("\n");
}

int main() {
    Node* poly1 = NULL;
    Node* poly2 = NULL;

    printf("Create first polynomial:\n");
    createPoly(&poly1);

    printf("Create second polynomial:\n");
    createPoly(&poly2);

    // Display input polynomials
    printf("First polynomial: ");
    displayPoly(poly1);

    printf("Second polynomial: ");
    displayPoly(poly2);

    // Add and display the sum of polynomials
    Node* result = addP(poly1, poly2);
    printf("Sum of polynomials: ");
    displayPoly(result);

    // Multiply and display the product of polynomials
    Node* product = multiP(poly1, poly2);
    printf("Product of polynomials: ");
    displayPoly(product);

    return 0;
}
