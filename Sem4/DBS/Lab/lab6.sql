/*
Student(Roll,GPA);
*/

dECLARE 
 grade CHAR(1); 
BEGIN 
 grade := '&g'; 
IF grade = 'A' THEN 
 DBMS_OUTPUT.PUT_LINE('Excellent'); 
ELSIF grade = 'B' THEN 
 DBMS_OUTPUT.PUT_LINE('Very Good'); 
ELSIF grade = 'C' THEN 
 DBMS_OUTPUT.PUT_LINE('Good'); 
ELSIF grade = 'D' THEN 
 DBMS_OUTPUT. PUT_LINE('Fair'); 
ELSIF grade = 'F' THEN 
 DBMS_OUTPUT.PUT_LINE('Poor'); 
ELSE 
 DBMS_OUTPUT.PUT_LINE('No such grade'); 
END IF; 
END; 
/ 

DECLARE 
 x NUMBER := 0; 
BEGIN 
 WHILE x < 4  
   LOOP 
  DBMS_OUTPUT.PUT_LINE ('Inside loop: x = ' || 
          TO_CHAR(x)); 
  X := x + 1; 
   END LOOP ;
END; 
/

CREATE TABLE account (
    account_number NUMBER PRIMARY KEY,
    balance NUMBER 
);

SET SERVEROUTPUT ON
DECLARE 
    B NUMBER;
BEGIN 
    SELECT COUNT(BALANCE) INTO B FROM account WHERE BALANCE < 0;
    IF B = 0 THEN RAISE NO_DATA_FOUND;
    END IF;
    UPDATE ACCO SET BALANCE = 0 WHERE BALANCE < 0;
EXCEPTION 
    WHEN NO_DATA_FOUND THEN DBMS_OUTPUT.PUT_LINE('NO ROWS FOUND');
END;
/

--OR

BEGIN
    UPDATE account SET balance = 0 WHERE balance < 0;

    IF SQL%ROWCOUNT = 0 THEN
        DBMS_OUTPUT.PUT_LINE('No rows updated');
    END IF;

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred');
END;
/

Declare 
 Insufficient_Balance Exception; 
 Amount account.balance%Type; 
 Temp account.balance%Type; 
 ANumber account.account_number%Type;  
BEGIN 
 ANumber := &Number; 
 Amount := &Amount; 
 Select balance into Temp from account where  
     account_number = ANumber; 
 Temp := Temp – Amount; 
 IF (Temp >=500) THEN 
 update account set balance = Temp where account_number =  
 ANumber; 
 ELSE 
 RAISE Insufficient_Balanae; 
 end if;
EXCEPTION 
 WHEN Insufficient_Balance THEN  
  dbms_output.put_line(“Insufficient Balance”); 
 
WHEN OTHERS THEN  
  dbms_output.put_line(“ERROR”); 
 
END;

CREATE TABLE Student (
    Roll NUMBER PRIMARY KEY,
    GPA NUMBER(3,1)
);

INSERT INTO Student (Roll, GPA) VALUES (1, 5.8);
INSERT INTO Student (Roll, GPA) VALUES (2, 6.5);
INSERT INTO Student (Roll, GPA) VALUES (3, 3.4);
INSERT INTO Student (Roll, GPA) VALUES (4, 7.8);
INSERT INTO Student (Roll, GPA) VALUES (5, 9.5);

set serveroutput on;

--1. Write a PL/SQL block to display the GPA of given student.
declare
    ro number:=&ro;
    gpa Student.GPA%type;
begin
    select GPA into gpa
    from Student
    where Roll=ro;

    DBMS_OUTPUT.PUT_LINE('GPA of student with Roll ' || ro || ' is: ' || gpa);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No student found with Roll ' || ro);
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred');
END;
/

--2. Write a PL/SQL block to display the letter grade(0-4: F; 4-5: E; 5-6: D; 6-7: C;                 
--7-8: B; 8-9: A; 9-10: A+} of given student.
DECLARE
    v_roll student.roll%TYPE;
    v_gpa student.gpa%TYPE;
    v_grade CHAR(2);
BEGIN
    v_roll := &RollNumber;

    SELECT gpa INTO v_gpa FROM student WHERE roll = v_roll;

    IF v_gpa BETWEEN 0 AND 4 THEN
        v_grade := 'F';
    ELSIF v_gpa BETWEEN 4 AND 5 THEN
        v_grade := 'E';
    ELSIF v_gpa BETWEEN 5 AND 6 THEN
        v_grade := 'D';
    ELSIF v_gpa BETWEEN 6 AND 7 THEN
        v_grade := 'C';
    ELSIF v_gpa BETWEEN 7 AND 8 THEN
        v_grade := 'B';
    ELSIF v_gpa BETWEEN 8 AND 9 THEN
        v_grade := 'A';
    ELSIF v_gpa BETWEEN 9 AND 10 THEN
        v_grade := 'A+';
    END IF;

    DBMS_OUTPUT.PUT_LINE('Student Roll Number: ' || v_roll);
    DBMS_OUTPUT.PUT_LINE('GPA: ' || v_gpa);
    DBMS_OUTPUT.PUT_LINE('Letter Grade: ' || v_grade);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Error: No student found with this roll number.');
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred.');
END;
/

--3.Input the date of issue and date of return for a book. Calculate and display the fine 
--with the appropriate message using a PL/SQL block. The fine is charged as per 
--the table 8.1:  
Late period 
7 days 
Fine 
8 – 15 days 
NIL 
Rs.1/day 
16 - 30 days 
After 30 days 
Rs. 2/ day 
Rs. 5.00 
Table 8.1

DECLARE
    v_issue_date DATE;
    v_return_date DATE;
    v_late_days NUMBER;
    v_fine NUMBER := 0;
BEGIN
    -- Input Dates
    v_issue_date := TO_DATE('&IssueDate', 'DD-MM-YYYY');
    v_return_date := TO_DATE('&ReturnDate', 'DD-MM-YYYY');

    -- Calculate the number of late days (if any)
    v_late_days := v_return_date - v_issue_date - 7; -- 7-day grace period

    IF v_late_days <= 0 THEN
        v_fine := 0; -- No fine within 7 days
    ELSIF v_late_days BETWEEN 1 AND 8 THEN
        v_fine := v_late_days * 1; -- Rs. 1/day for 8-15 days late
    ELSIF v_late_days BETWEEN 9 AND 23 THEN
        v_fine := v_late_days * 2; -- Rs. 2/day for 16-30 days late
    ELSIF v_late_days > 23 THEN
        v_fine := v_late_days * 5; -- Rs. 5/day after 30 days
    END IF;

    -- Display Results
    DBMS_OUTPUT.PUT_LINE('Date of Issue: ' || TO_CHAR(v_issue_date, 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Date of Return: ' || TO_CHAR(v_return_date, 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Late Days: ' || GREATEST(v_late_days, 0));
    DBMS_OUTPUT.PUT_LINE('Total Fine: Rs. ' || v_fine);

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred. Please check the input dates.');
END;
/
