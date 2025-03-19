TRIGGERS
-- 1) Based on the University database Schema in Lab 2, write a row trigger that records along with the time any change made in the Takes (ID, course-id, sec-id,semester, year, grade) table in log_change_Takes (Time_Of_Change, ID, courseid,sec-id, semester, year, grade
CREATE TABLE log_change_takes(
 time_of_change TIMESTAMP(2),
 ID varchar(5),
 course_id varchar(8),
 sec_id varchar(8),
 semester varchar(6),
 year number(4, 0),
 grade varchar(2)
);

CREATE OR REPLACE TRIGGER log_takes
AFTER UPDATE ON takes
FOR EACH ROW
BEGIN
 INSERT INTO log_change_takes VALUES(CURRENT_TIMESTAMP, :OLD.ID, :OLD.course_id,
:OLD.sec_id, :OLD.semester, :OLD.year, :OLD.grade);
END;
/


-- 2 
-- Based on the University database schema in Lab: 2, write a row trigger to insert 
-- the existing values of the Instructor (ID, name, dept-name, salary) table into a new
-- table Old_ Data_Instructor (ID, name, dept-name, salary) when the salary table is updated. 
CREATE TABLE old_data_instructor(
 ID varchar(5),
 name varchar(20),
 dept_name varchar(20),
 salary numeric(8, 2), check (salary > 29000)
);
CREATE OR REPLACE TRIGGER sal_trigger
AFTER UPDATE ON instructor
FOR EACH ROW
BEGIN
 INSERT INTO old_data_instructor VALUES(:OLD.ID, :OLD.name, :OLD.dept_name,
:OLD.salary);
END;
/
3

CREATE OR REPLACE TRIGGER Instr_check
BEFORE INSERT ON instructor
FOR EACH ROW

DECLARE
    dept_budget NUMBER;

BEGIN
    -- Fetch the department budget
    SELECT budget INTO dept_budget
    FROM Department
    WHERE Department.dept_name = :NEW.dept_name;

    -- Check salary constraints and name validation
    IF :NEW.salary > 0 AND :NEW.salary <= dept_budget AND REGEXP_LIKE(:NEW.name, '^[a-zA-Z]+$') THEN
        dbms_output.put_line('Valid salary and name');
    ELSE
        RAISE_APPLICATION_ERROR(-20001, 'Invalid salary: Either zero, negative, exceeds department budget, or name is not alphabetic.');
    END IF;
END;
/

4

create table Client_Master(
	client_no number primary key,
	name varchar(20),
	address varchar(50),
	bal_due number);
create table auditclient(
	client_no number primary key,
	name varchar(20),
	bal_due number,
	operation varchar(10) check(operation in('update','delete')),
	userid number,
	opdate timestamp);
create or replace trigger audit_system1
after update on client_master
for each row
begin
insert into auditclient values(:new.client_no,:new.name,:new.bal_due,'update',1234,current_timestamp);
end;
/
create or replace trigger audit_system2
after delete on client_master
for each row
begin
insert into auditclient values(:old.client_no,:old.name,:old.bal_due,'delete',1234,current_timestamp);
end;
/


5


create view advisor_student as
select S.name as student_name,S.dept_name as student_dept,S.tot_cred,A.s_id,A.i_id,I.name as instr_name,I.dept_name as instr_dept,salary
from advisor A
join student S on S.id = A.s_id
join instructor I on I.id = A.I_id;

create or replace trigger delete_advisor_student
instead of delete on advisor_student 
for each row
begin
delete from advisor where s_id = :old.s_id and i_id = :old.i_id;
dbms_output.put_line('trigger successfully deleted info');
end;
/

delete from advisor_student where s_id = 12345 and i_id = 10101;

