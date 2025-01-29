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

--Implement the following Queries on UNIVERSITY Database:

--Group By:
--1. Find the number of students in each course. 
select course_id, count(distinct id) as "Number of Students"
from takes
group by course_id;

--Wrong ->

select title,count(id)
from student,course
where student.dept_name = course.dept_name
group by title;

--*2. Find those departments where the average number of students are greater than 10. 
SELECT dept_name, AVG(count) AS avg_count
FROM (
    SELECT c.dept_name, COUNT(t.id) AS count
    FROM takes t,course c
    where t.course_id = c.course_id
    GROUP BY t.course_id
) temp
GROUP BY dept_name
HAVING AVG(count) > 10;

QUESTION CORRECTED -
--2. Find those departments where the number of students are greater than 10.
select dept_name,count(id) 
from student
group by dept_name
having count(id)>10;

--3. Find the total number of courses in each department. 
select dept_name,count(title)
from course
group by dept_name;

--4. Find the names and average salaries of all departments whose average salary is 
--greater than 42000. 
select dept_name,avg(salary)
from instructor
group by dept_name
having avg(salary)>42000;

--5. Find the enrolment of each section that was offered in Spring 2009. 
select sec_id,count(id)
from takes
where semester = 'Spring' and year = 2009
group by sec_id;

--Ordering the display of Tuples (Use ORDER BY ASC/DESC):

--6. List all the courses with prerequisite courses, then display course id in increasing 
--order.
select *
from prereq
order by course_id;

--7. Display the details of instructors sorting the salary in decreasing order.
select *
from instructor
order by salary desc;

--Derived Relations: 

--8. Find the maximum total salary across the departments. 
select max(sum(salary)) as "Max Salary"
from instructor
group by dept_name;

with max_sum as(
    select max(sum(salary)) as max
    from instructor
    group by dept_name
)
select dept_name,sum(salary) as max_salary
from instructor
group by dept_name
having sum(salary) = (
    select max
    from max_sum
);

--**9. Find the average instructors’ salaries of those departments where the average 
--salary is greater than 42000. 
select avg(salary) as avg_salary
from instructor
where dept_name in(
    select dept_name
    from instructor
    group by dept_name
    having avg(salary)>42000
);

--10. Find the sections that had the maximum enrolment in Spring 2010 
select sec_id,count(id)
from takes
where semester='Spring' and year=2010
group by sec_id,course_id
having count(id) in(
    select max(count(id))
    from takes
    where semester='Spring' and year=2010
    group by sec_id,course_id
);

--*11. Find the names of all instructors who teach all students that belong to ‘CSE’ 
--department.
select name
from instructor i
where not exists(
    select id
    from student
    where dept_name='Comp. Sci.'
    minus
    select t1.id
    from takes t1,teaches t2
    where t1.course_id=t2.course_id and t1.sec_id=t2.sec_id 
        and t1.semester=t2.semester and t1.year=t2.year 
        and t2.id=i.id
);

--12. Find the average salary of those department where the average salary is greater 
--than 50000 and total number of instructors in the department are more than 5.
select dept_name,avg(Salary) as avg_salary
from instructor
group by dept_name
having avg(salary)>50000 and count(id)>5;

--With Clause:

--13. Find all departments with the maximum budget.
with dept_total_table as(
    select dept_name,sum(salary) as total_salary
    from instructor
    group by dept_name
)
select dept_name
from dept_total_table
group by dept_name
having total_salary = max(total_salary);

--14. Find all departments where the total salary is greater than the average of the total 
--salary at all departments.
with dept_avg_table as(
    select avg(salary) as avg_salary
    from instructor
    group by dept_name
)
select dept_name,sum(salary) as total_salary
from instructor
group by dept_name
having sum(salary) > all(
    select avg_salary
    from dept_avg_table
);

--(Use ROLLBACK (and SAVEPOINT) to undo the effect of any modification on database before COMMIT)
--15. Transfer all the students from CSE department to IT department.

--16. Increase salaries of instructors whose salary is over $100,000 by 3%, and all
--others receive a 5% raise
