# 1. Create Employee table with following constraints:
```sql
-- Make EmpNo as Primary key. 
-- Do not allow EmpName, Gender, Salary and Address to have null values. 
-- Allow Gender to have one of the two values: ‘M’, ‘F’. 
CREATE TABLE employee(
    empno NUMBER(5) PRIMARY KEY,
    empname VARCHAR(20) NOT NULL,
    gender CHAR(1) CHECK (gender IN ('M', 'F')) NOT NULL,
    salary NUMBER(10,2) NOT NULL,
    address VARCHAR(50) NOT NULL,
    dno NUMBER(5)
);
```

# 2. Create Department table with following:
```sql
-- Make DeptNo as Primary key 
-- Make DeptName as candidate key 
CREATE TABLE department(
    deptno NUMBER(5) PRIMARY KEY,
    deptname VARCHAR(20) UNIQUE,
    location VARCHAR(20)
);
```

# 3. Make DNo of Employee as foreign key which refers to DeptNo of Department.
```sql
ALTER TABLE employee
ADD CONSTRAINT fk_dno FOREIGN KEY (dno)
REFERENCES department (deptno);
```

# 4. Insert few tuples into Employee and Department which satisfies the above constraints.
```sql
INSERT INTO department VALUES (1,'IT','Bangalore');
INSERT INTO department VALUES (2,'HR','Mysore');
INSERT INTO department VALUES (3,'Finance','Mangalore');
INSERT INTO department VALUES (4,'Admin','Chennai');
INSERT INTO department VALUES (5,'Sales','Hyderabad');
INSERT INTO employee VALUES (1,'John','M',1000,'Bangalore',1);
INSERT INTO employee VALUES (2,'Doe','F',2000,'Mysore',2);
INSERT INTO employee VALUES (3,'Jane','F',3000,'Mangalore',3);
INSERT INTO employee VALUES (4,'Dane','M',4000,'Chennai',4);
INSERT INTO employee VALUES (5,'Dane','M',5000,'Hyderabad',5);
```

# 5. Try to insert few tuples into Employee and Department which violates some of the above constraints.
```sql
INSERT INTO department VALUES (1,'IT','Bangalore');
--ORA-00001: unique constraint violated
INSERT INTO employee VALUES (1,'John','M',1000,'Bangalore',1);
--ORA-00001: unique constraint violated
INSERT INTO employee VALUES (6,'Dane','M',5000,'Hyderabad',5);
--ORA-02291: integrity constraint violated - parent key not found
INSERT INTO employee VALUES (7,'Dane','M',5000,'Hyderabad',6);
--ORA-02291: integrity constraint violated - parent key not found
```

# 6. Try to modify/delete a tuple which violates a constraint.
```sql
--(Ex: Drop a department tuple which has one or more employees) 
DELETE FROM department WHERE deptno=1;
--ORA-02292: integrity constraint violated - parent key must
--be found in the table
```

# 7. Modify the foreign key constraint of Employee table such that whenever a department tuple is deleted, the employees belonging to that department will also be deleted.
```sql
ALTER TABLE employee
DROP CONSTRAINT fk_dno;
ALTER TABLE employee
ADD CONSTRAINT fk_dno FOREIGN KEY (dno) 
REFERENCES department (deptno) ON DELETE CASCADE;
```

# Naming Constraints:

## 8. Create a named constraint to set the default salary to 10000 and test the constraint by inserting a new record.
```sql
ALTER TABLE employee
ADD CONSTRAINT default_salary DEFAULT 10000 FOR salary;
```

# University Database Schema Queries:

## 9. List all Students with names and their department names.
```sql
SELECT name, dept_name FROM student;
```

## 10. List all instructors in Comp. Sci. department.
```sql
SELECT name
FROM instructor 
WHERE dept_name='Comp. Sci.';
```

## 11. Find the names of courses in Comp. Sci. department which have 3 credits.
```sql
SELECT title
FROM course
WHERE dept_name='Comp. Sci.' AND credits=3;
```

## 12. For the student with ID 12345 (or any other value), show all course-id and title of all courses registered for by the student.
```sql
SELECT course.course_id, title
FROM takes, course
WHERE id='12345' AND takes.course_id=course.course_id;
```

## 13. List all the instructors whose salary is in between 40000 and 90000.
```sql
SELECT name
FROM instructor
WHERE salary BETWEEN 40000 AND 90000;
```

## 14. Display the IDs of all instructors who have never taught a course.
```sql
SELECT id
FROM instructor
WHERE id NOT IN (SELECT id FROM teaches);
```

## 15. Find the student names, course names, and the year, for all students those who have attended classes in room-number 101.
```sql
SELECT student.name, course.title, section.year
FROM student, takes, section, course
WHERE student.id=takes.id AND takes.course_id=course.course_id
    AND takes.sec_id=section.sec_id AND section.room_number=101
    AND takes.year=section.year AND takes.course_id=section.course_id;
```

## 16. For all students who have opted courses in 2010, find their names and course id’s with the attribute course title replaced by c_name.
```sql
SELECT student.name, course.course_id, course.title AS c_name
FROM student, takes, course
WHERE takes.year='2010' AND student.id=takes.id
    AND takes.course_id=course.course_id;
```

## 17. Find the names of all instructors whose salary is greater than the salary of at least one instructor of Comp. Sci. department and salary replaced by inst_salary.
```sql
SELECT DISTINCT t.name, t.salary inst_salary
FROM instructor t, instructor s
WHERE t.salary > s.salary AND s.dept_name='Comp. Sci.';
```

## 18. Find the names of all instructors whose department name includes the substring ‘e’.
```sql
SELECT name
FROM instructor
WHERE name LIKE '%e%';
```

## 19. List the student names along with the length of the student names.
```sql
SELECT name, LENGTH(name) AS name_length
FROM student;
```

## 20. List the department names and 3 characters from 3rd position of each department name.
```sql
SELECT dept_name, SUBSTR(dept_name, 3, 3)
FROM department;
```

## 21. List the instructor names in upper case.
```sql
SELECT UPPER(name)
FROM instructor;
```

## 22. Replace NULL with value1(say 0) for a column in any of the table.
```sql
SELECT NVL(salary, 0) AS salary
FROM instructor;
```

## 23. Display the salary and salary/3 rounded to nearest hundred from Instructor.
```sql
SELECT salary, ROUND(salary/300, 0) * 100 AS salary_rounded
FROM instructor;
```

## 24. Display the birth date of all the employees in the following formats:
```sql
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MON-YYYY') AS DOB_FORMAT_1
FROM employee;
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MON-YY') AS DOB_FORMAT_2
FROM employee;
SELECT EMPNO, EMPNAME, TO_CHAR(DOB, 'DD-MM-YY') AS DOB_FORMAT_3
FROM employee;
```

## 25. List the employee names and the year (fully spelled out) in which they are born:
```sql
SELECT EMPNAME, TO_CHAR(DOB, 'YYYY') AS "YEAR"
FROM employee;
SELECT EMPNAME, INITCAP(TO_CHAR(DOB, 'YYYY')) AS "Year"
FROM employee;
SELECT EMPNAME, LOWER(TO_CHAR(DOB, 'YYYY')) AS "year"
FROM employee;
```
