#include <unistd.h>
#include <stdio.h>
#include <string.h>

int main() {
    int a = -42;
    unsigned int b = 42;
    float c = 3.14159;
    char d = 'A';
    char str[] = "Hello";

    char buf[100];

    // Integer
    snprintf(buf, sizeof(buf), "Integer (%%d): %d\n", a);
    write(1, buf, strlen(buf));

    // Unsigned
    snprintf(buf, sizeof(buf), "Unsigned (%%u): %u\n", b);
    write(1, buf, strlen(buf));

    // Float
    snprintf(buf, sizeof(buf), "Float (%%f): %f\n", c);
    write(1, buf, strlen(buf));

    // Character
    snprintf(buf, sizeof(buf), "Character (%%c): %c\n", d);
    write(1, buf, strlen(buf));

    // String
    snprintf(buf, sizeof(buf), "String (%%s): %s\n", str);
    write(1, buf, strlen(buf));

    // Hexadecimal
    snprintf(buf, sizeof(buf), "Hexadecimal (%%x): %x\n", b);
    write(1, buf, strlen(buf));

    // Octal
    snprintf(buf, sizeof(buf), "Octal (%%o): %o\n", b);
    write(1, buf, strlen(buf));

    // Percent sign
    snprintf(buf, sizeof(buf), "Percentage (%%%%): %%\n");
    write(1, buf, strlen(buf));

    return 0;
}
