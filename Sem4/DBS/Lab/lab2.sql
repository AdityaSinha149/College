/*
Consider the following schema: 
Employee (EmpNo, EmpName, Gender, Salary, Address, DNo) 
Department (DeptNo, DeptName, Location) 
*/

--1. Create Employee table with following constraints: 
-- Make EmpNo as Primary key. 
-- Do not allow EmpName, Gender, Salary and Address to have null values. 
-- Allow Gender to have one of the two values: ‘M’, ‘F’. 
create table employee(
    empno number(5) primary key,
    empname varchar(20),
    gender char(1) check (gender in ('M', 'F')) not null,
    salary number(10,2) not null,
    address varchar(50) not null,
    dno number(5)
);

--2. Create Department table with following: 
-- Make DeptNo as Primary key 
-- Make DeptName as candidate key 
create table studepartment(
    deptno number(5) primary key,
    deptname varchar(20) unique,
    location varchar(20)
);

--3. Make DNo of Employee as foreign key which refers to DeptNo of Department.
alter table employee
add constraint fk_dno foreign key (dno)
references studepartment (deptno);

--4. Insert few tuples into Employee and Department which satisfies the above constraints. 
insert into department values(1,'IT','Bangalore');
insert into department values(2,'HR','Mysore');
insert into department values(3,'Finance','Mangalore');
insert into department values(4,'Admin','Chennai');
insert into department values(5,'Sales','Hyderabad');
insert into employee values(1,'John','M',1000,'Bangalore',1);
insert into employee values(2,'Doe','F',2000,'Mysore',2);
insert into employee values(3,'Jane','F',3000,'Mangalore',3);
insert into employee values(4,'Dane','M',4000,'Chennai',4);
insert into employee values(5,'Dane','M',5000,'Hyderabad',5);

--5. Try to insert few tuples into Employee and Department which violates some of the 
--above constraints. 
insert into department values(1,'IT','Bangalore');
--ORA-00001: unique constraint (SYSTEM.SYS_C0010001) violated
insert into employee values(1,'John','M',1000,'Bangalore',1);
--ORA-00001: unique constraint (SYSTEM.SYS_C0010000) violated
insert into employee values(6,'Dane','M',5000,'Hyderabad',5);
--ORA-02291: integrity constraint (SYSTEM.FK_DNO) violated - parent key not found
insert into employee values(7,'Dane','M',5000,'Hyderabad',6);
--ORA-02291: integrity constraint (SYSTEM.FK_DNO) violated - parent key not found

--6. Try to modify/delete a tuple which violates a constraint. 
--(Ex: Drop a department tuple which has one or more employees) 
delete from department where deptno=1;
--ORA-02292: integrity constraint (SYSTEM.FK_DNO) violated - parent key must
--be found in the table

--7. Modify the foreign key constraint of Employee table such that whenever a department 
--tuple is deleted, the employees belonging to that department will also be deleted. 
alter table employee
add constraint fk_dno foreign key(dno) 
references department(deptno) on delete cascade;

--Naming Constraints: 
--8. Create a named constraint to set the default salary to 10000 and test the constraint by  
--inserting a new record.
alter table employee
add constraint default_salary default 10000 for salary;

/*
University Database Schema: 
Student (ID, name,dept-name, tot-cred) 
Instructor(ID, name, dept-name, salary) 
Course (Course-id, title,dept-name, credits) 
Takes (ID, course-id, sec-id, semester, year, grade) 
Classroom (building, room-number, capacity) 
Department (dept-name, building, budget) 
Section (course-id, section-id, semester, year, building, room-number, time-slot-id) 
Teaches (id, course-id, section-id, semester, year) 
Advisor(s-id, i-id) 
Time-slot (time-slot-id, day, start-time, end-time) 
Prereq (course-id, prereq-id) 
*/

--9. List all Students with names and their department names. 
select name,dept_name from student;

--10. List all instructors in Comp. Sci. department. 
select name
from instructor 
where dept_name='Comp. Sci.';

--11. Find the names of courses in Comp. Sci. department which have 3 credits. 
select title
from course
where dept_name='Comp. Sci.' and credits=3;

--12. For the student with ID 12345 (or any other value), 
--show all course-id and title of all courses registered for by the student.
select course.course_id,title
from takes,course
where id='12345' and takes.course_id=course.course_id;

--13. List all the instructors whose salary is in between 40000 and 90000. 
select name
from instructor
where salary between 40000 and 90000;

--14. Display the IDs of all instructors who have never taught a course. 
select id
from instructor
where id not in (select id from teaches);

--15. Find the student names, course names, and the year, for all students those who have 
--attended classes in room-number 101. 
select student.name,course.title,section.year
from student,takes,section,course
where student.id=takes.id and takes.course_id=course.course_id
    and takes.sec_id=section.sec_id and section.room_number=101
    and takes.year=section.year and takes.course_id=section.course_id;

--16. For all students who have opted courses in 2010, find their names and course id’s 
--with the attribute course title replaced by c_name.
select student.name,course.course_id,course.title as c_name
from student,takes,course
where takes.year='2010' and student.id=takes.id
    and takes.course_id=course.course_id;

--17. Find the names of all instructors whose salary is greater than the salary of at least 
--one instructor of Comp. Sci. department and salary replaced by inst_salary. 
select distinct t.name,t.salary inst_salary
from instructor t,instructor s
where t.salary>s.salary and s.dept_name='Comp. Sci.';

--18. Find the names of all instructors whose department name includes the substring ‘e’.   
select name
from instructor
where name like '%e%';

--19. List the student names along with the length of the student names. 
select name,length(name) as name_length
from student;

--20. List the department names and 3 characters from 3rd position of each department name 
select dept_name,substr(dept_name,3,3)
from department;

--21. List the instructor names in upper case. 
select upper(name)
from instructor;

--22. Replace NULL with value1(say 0) for a column in any of the table 
select NVL(salary,0) as salary
from instructor;

--23. Display the salary and salary/3 rounded to nearest hundred from Instructor. 
select salary,round(salary/300,0)*100 as salary_rounded
from instructor;

--Add date of birth column (DOB) to Employee Table. Insert appropriate DOB values 
--for different employees and try the exercise problems given below: 
alter table employee
add dob varchar(10);

UPDATE employee SET DOB = TO_DATE('1985-03-25', 'YYYY-MM-DD') WHERE EMPNO = 1;
UPDATE employee SET DOB = TO_DATE('1990-07-12', 'YYYY-MM-DD') WHERE EMPNO = 2;
UPDATE employee SET DOB = TO_DATE('1987-11-05', 'YYYY-MM-DD') WHERE EMPNO = 3;
UPDATE employee SET DOB = TO_DATE('1992-01-22', 'YYYY-MM-DD') WHERE EMPNO = 4;
UPDATE employee SET DOB = TO_DATE('1989-09-09', 'YYYY-MM-DD') WHERE EMPNO = 5;
UPDATE employee SET DOB = TO_DATE('1988-06-15', 'YYYY-MM-DD') WHERE EMPNO = 6;
UPDATE employee SET DOB = TO_DATE('1991-12-30', 'YYYY-MM-DD') WHERE EMPNO = 7;



--24. Display the birth date of all the employees in the following format: 
-- ‘DD-MON-YYYY’ 
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MON-YYYY') AS DOB
FROM employee;

-- ‘DD-MON-YY’ 
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MON-YY') AS DOB
FROM employee;

-- ‘DD-MM-YY’ 
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MM-YY') AS DOB
FROM employee;

--25. List the employee names and the year (fully spelled out) in which they are born 
‘YEAR’
 SELECT EMPNAME, TO_CHAR(DOB, 'YYYY') AS YEAR FROM employee;

‘Year’ 
‘year’