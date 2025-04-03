#ifndef LCD_H
#define LCD_H

#include <LPC17xx.h>
#include <stdlib.h>

void delay(unsigned int count) {
    volatile unsigned int r;
    for (r = 0; r < count; r++);
}

void lcd_cmd(int cmd) {
    LPC_GPIO0->FIOCLR = (0x0F << 23) | (1 << 27) | (1 << 28);
    LPC_GPIO0->FIOPIN = (cmd & 0xF0) << 19;
    LPC_GPIO0->FIOCLR = 1 << 27;
    LPC_GPIO0->FIOSET = 1 << 28;
    delay(50);
    LPC_GPIO0->FIOCLR = 1 << 28;
    LPC_GPIO0->FIOPIN = (cmd & 0x0F) << 23;
    LPC_GPIO0->FIOSET = 1 << 28;
    delay(50);
    LPC_GPIO0->FIOCLR = 1 << 28;
}

void lcd_data(char data) {
    LPC_GPIO0->FIOCLR = (0x0F << 23) | (1 << 27) | (1 << 28);
    LPC_GPIO0->FIOPIN = (data & 0xF0) << 19;
    LPC_GPIO0->FIOSET = 1 << 27;
    LPC_GPIO0->FIOSET = 1 << 28;
    delay(50);
    LPC_GPIO0->FIOCLR = 1 << 28;
    LPC_GPIO0->FIOPIN = (data & 0x0F) << 23;
    LPC_GPIO0->FIOSET = 1 << 28;
    delay(50);
    LPC_GPIO0->FIOCLR = 1 << 28;
}

void lcd_init(void) {
    LPC_PINCON->PINSEL1 &= ~(0x3FFC << 12);
    LPC_GPIO0->FIODIR |= (0x0F << 23) | (1 << 27) | (1 << 28);
    delay(15000);
    lcd_cmd(0x33);
    lcd_cmd(0x32);
    lcd_cmd(0x28);
    lcd_cmd(0x0C);
    lcd_cmd(0x06);
    lcd_cmd(0x01);
}

void lcd_puts(const char *str) {
    while (*str) {
        lcd_data(*str++);
    }
}

#endif
