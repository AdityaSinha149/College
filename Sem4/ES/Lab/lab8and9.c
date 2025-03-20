	//lab-8
		
		
//#include <LPC17xx.h>
//#include <stdlib.h>

//unsigned char msg1[13] = "Dice Result"; // Message to display initially
//unsigned char key;
//unsigned long int temp1 = 0;

// Function prototypes for LCD
//void lcd_init(void);
//void write(int, int);
//void delay_lcd(unsigned int);
//void lcd_comdata(int, int);
//void clear_ports(void);
//void lcd_puts(unsigned char *);

//void lcd_init() {
    /* Ports initialized as GPIO */
  //  LPC_PINCON->PINSEL1 &= 0xFC003FFF; // P0.23 to P0.28
    /* Setting the directions as output */
    //LPC_GPIO0->FIODIR |= 0x0F << 23 | 1 << 27 | 1 << 28;
    //clear_ports();
    //delay_lcd(3200);
    //lcd_comdata(0x33, 0);
    //delay_lcd(30000);
    //lcd_comdata(0x32, 0);
    //delay_lcd(30000);
    //lcd_comdata(0x28, 0); // function set
    //delay_lcd(30000);
    //lcd_comdata(0x0c, 0); // display on cursor off
    //delay_lcd(800);
    //lcd_comdata(0x06, 0); // entry mode set increment cursor right
    //delay_lcd(800);
    //lcd_comdata(0x01, 0); // display clear
    //delay_lcd(10000);
//}

//void lcd_comdata(int temp1, int type) {
  //  int temp2 = temp1 & 0xf0; // move data (26-8+1) times : 26 - HN place, 4 - Bits
    //temp2 = temp2 << 19; // data lines from 23 to 26
    //write(temp2, type);
    //temp2 = temp1 & 0x0f; // 26-4+1
    //temp2 = temp2 << 23;
    //write(temp2, type);
    //delay_lcd(1000);
//}

//void write(int temp2, int type) { /* write to command/data reg */
  //  clear_ports();
    //LPC_GPIO0->FIOPIN = temp2; // Assign the value to the data lines
    //if (type == 0)
      //  LPC_GPIO0->FIOCLR = 1 << 27; // clear bit RS for Command
    //else
      //  LPC_GPIO0->FIOSET = 1 << 27; // set bit RS for Data
    //LPC_GPIO0->FIOSET = 1 << 28; // EN = 1
    //delay_lcd(25);
    //LPC_GPIO0->FIOCLR = 1 << 28; // EN = 0
//}

//void delay_lcd(unsigned int r1) {
  //  unsigned int r;
    //for (r = 0; r < r1; r++);
//}

//void clear_ports(void) { /* Clearing the lines at power on */
  //  LPC_GPIO0->FIOCLR = 0x0F << 23; // Clearing data lines
    //LPC_GPIO0->FIOCLR = 1 << 27; // Clearing RS line
    //LPC_GPIO0->FIOCLR = 1 << 28; // Clearing Enable line
//}

//void lcd_puts(unsigned char *buf1) {
  //  unsigned int i = 0;
    //unsigned int temp3;
    //while (buf1[i] != '\0') {
      //  temp3 = buf1[i];
        //lcd_comdata(temp3, 1);
        //i++;
        //if (i == 16)
          //  lcd_comdata(0xc0, 0); // move to the next line
    //}
//}

//int main() {
  //  unsigned char k;
    //lcd_init(); // Initialize LCD
    //temp1 = 0x80;
    //lcd_comdata(temp1, 0); // Set the cursor to the start
    //delay_lcd(800);
    //lcd_puts(&msg1[0]); // Display initial message
    
    //while (1) {
      //  if (!(LPC_GPIO2->FIOPIN & 1 << 12)) { // Check if the button is pressed
        //    k = (rand() % 6) + 1; // Generate a random number between 1 and 6
          //  k = k + 0x30; // Convert to ASCII character ('1' to '6')
            //temp1 = 0xc0; // Move the cursor to the second line
            //lcd_comdata(temp1, 0); 
            //delay_lcd(800);
            //lcd_puts(&k); // Display the dice result on the LCD
        //}
    //}
//}


//lab-9

#include <LPC17xx.h>
#include <stdlib.h>
void lcd_init(void);
void write(int, int);
void delay_lcd(unsigned int);
void lcd_comdata(int, int); 
void clear_ports(void);
void lcd_puts(unsigned char *);

void lcd_init() {
	/*Ports initialized as GPIO */
	LPC_PINCON->PINSEL1 &= 0xFC003FFF; //P0.23 to P0.28
	/*Setting the directions as output */
	LPC_GPIO0->FIODIR |= 0x0F<<23 | 1<<27 | 1<<28;
	clear_ports();
	delay_lcd(3200);
	lcd_comdata(0x33, 0); 
	delay_lcd(30000);
	lcd_comdata(0x32, 0);
	delay_lcd(30000);
	lcd_comdata(0x28, 0); //function set
	delay_lcd(30000);
	lcd_comdata(0x0c, 0);//display on cursor off
	delay_lcd(800);
	lcd_comdata(0x06, 0); //entry mode set increment cursor right
	delay_lcd(800);
	lcd_comdata(0x01, 0); //display clear
	delay_lcd(10000);
	return;
}
void lcd_comdata(int temp1, int type) {
	int temp2 = temp1 & 0xf0; //move data (26-8+1) times : 26 - HN place, 4 - Bits
	temp2 = temp2 << 19; //data lines from 23 to 26
	write(temp2, type);
	temp2 = temp1 & 0x0f; //26-4+1
	temp2 = temp2 << 23; 
	write(temp2, type);
	delay_lcd(1000);
	return;
}
void write(int temp2, int type) { /*write to command/data reg */
	clear_ports();
	LPC_GPIO0->FIOPIN = temp2; // Assign the value to the data lines 
	if(type==0)
		LPC_GPIO0->FIOCLR = 1<<27; // clear bit RS for Command
	else
		LPC_GPIO0->FIOSET = 1<<27; // set bit RS for Data
	LPC_GPIO0->FIOSET = 1<<28; // EN=1
	delay_lcd(25);
	LPC_GPIO0->FIOCLR = 1<<28; // EN =0
	return;
}
void delay_lcd(unsigned int r1)
{
 unsigned int r;
 for(r=0;r<r1;r++);
 return;
}
void clear_ports(void) { /* Clearing the lines at power on */
	LPC_GPIO0->FIOCLR = 0x0F<<23; //Clearing data lines
	LPC_GPIO0->FIOCLR = 1<<27; //Clearing RS line
	LPC_GPIO0->FIOCLR = 1<<28; //Clearing Enable line
	return;
}
void lcd_puts(unsigned char *buf1) {
	unsigned int i=0;
	unsigned int temp3;
	while(buf1[i]!='\0') {
		temp3 = buf1[i];
		lcd_comdata(temp3, 1);
		i++;
		if(i==16)
			lcd_comdata(0xc0, 0);
	}
	return;
}

void scan(void);

unsigned char row, flag, key;
unsigned long int i, var1, temp, temp1 = 0, temp2, temp3;
unsigned char scan_code[16]={0x11,0x21,0x41,0x81,0x12,0x22,0x42,0x82,
	0x14,0x24,0x44,0x84,0x18,0x28,0x48,0x88};
unsigned char ascii_code[16]={'0','1','2','3','4','5',
	'6','7','8','9','A','B','+','-','*','/'};
int idx = 1;
int ans = 0;
unsigned char finans[3] = {'0','0','\0'};
int a = 0;
int b = 0;
char op;
int count = 0;

int main(void)
{
	LPC_GPIO2->FIODIR = 0x3c00;
	LPC_GPIO1->FIODIR = 0xf87fffff;
	lcd_init();
	temp1 = 0x80;
	lcd_comdata(temp1,0);
	delay_lcd(80000);
	while(count<3)
	{
		while(1)
		{
			for(row=1; row<5; row++)
			{
				if(row==1)
					var1 = 0x400;
				else if(row==2)
					var1 = 0x800;
				else if(row==3)
					var1 = 0x1000;
				else if(row==4)
					var1 = 0x2000;
				temp = var1;
				LPC_GPIO2->FIOCLR = 0x3c00;
				LPC_GPIO2->FIOSET = var1;
				flag = 0;
				scan();
				if(flag==1)
				{
					count++;
					break;
				}
			}
			if(flag==1)
				break;
		}
		for(i=0; i<16; i++)
		{
			if(key==scan_code[i])
			{
				key = ascii_code[i];
				lcd_puts(&key);
				delay_lcd(100000);
				if(count==1)
					a = key-48;
				else if(count==2)
					op = key;
				else if(count==3)
					b = key-48;
				break;
			}
		}
	}
	temp1 = 0xc0;
	lcd_comdata(temp1,0);
	delay_lcd(800);
	switch(op)
	{
		case '+':
			ans = a+b;
			break;
		case '-':
			ans = a-b;
			break;
		case '*':
			ans = a*b;
			break;
		case '/':
			ans = a/b;
			break;
	}
	while(ans!=0)
	{
		finans[idx--] = (ans%10)+48;
		ans = ans/10;
	}
	lcd_puts(&finans[0]);
	return 0;
}

void scan(void)
{
	temp3 = LPC_GPIO1->FIOPIN;
	temp3 &= 0x07800000;
	if(temp3!=0)
	{
		flag = 1;
		temp3>>=19;
		temp>>=10;
		key = temp3|temp;
	}
}	