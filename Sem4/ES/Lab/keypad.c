#include "lcd.h"
#include <LPC17xx.h>

void scan(void);

unsigned char row, flag, key;
unsigned long int i, var1, temp, temp1 = 0, temp2, temp3;
unsigned char scan_code[16] = {0x11, 0x21, 0x41, 0x81, 0x12, 0x22, 0x42, 0x82,
                               0x14, 0x24, 0x44, 0x84, 0x18, 0x28, 0x48, 0x88};
unsigned char ascii_code[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', '+', '-', '*', '/'};
int idx = 1, ans = 0, a = 0, b = 0, count = 0;
char op;
unsigned char finans[3] = {'0', '0', '\0'};

int main(void)
{
    LPC_GPIO2->FIODIR = 0x3c00;
    LPC_GPIO1->FIODIR = 0xf87fffff;
    lcd_init();
    lcd_comdata(0x80, 0);
    delay_lcd(80000);
    
    while (count < 3)
    {
        while (1)
        {
            for (row = 1; row < 5; row++)
            {
                var1 = (row == 1) ? 0x400 : (row == 2) ? 0x800 : (row == 3) ? 0x1000 : 0x2000;
                LPC_GPIO2->FIOCLR = 0x3c00;
                LPC_GPIO2->FIOSET = var1;
                flag = 0;
                scan();
                if (flag == 1)
                {
                    count++;
                    break;
                }
            }
            if (flag == 1)
                break;
        }
        for (i = 0; i < 16; i++)
        {
            if (key == scan_code[i])
            {
                key = ascii_code[i];
                lcd_puts(&key);
                delay_lcd(100000);
                if (count == 1)
                    a = key - '0';
                else if (count == 2)
                    op = key;
                else if (count == 3)
                    b = key - '0';
                break;
            }
        }
    }
    lcd_comdata(0xC0, 0);
    delay_lcd(800);
    
    switch (op)
    {
    case '+':
        ans = a + b;
        break;
    case '-':
        ans = a - b;
        break;
    case '*':
        ans = a * b;
        break;
    case '/':
        ans = a / b;
        break;
    }
    
    while (ans != 0)
    {
        finans[idx--] = (ans % 10) + '0';
        ans /= 10;
    }
    lcd_puts(finans);
    return 0;
}

void scan(void)
{
    temp3 = LPC_GPIO1->FIOPIN & 0x07800000;
    if (temp3 != 0)
    {
        flag = 1;
        temp3 >>= 19;
        temp >>= 10;
        key = temp3 | temp;
    }
}
