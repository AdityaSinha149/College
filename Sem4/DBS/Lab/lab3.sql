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

--Set Operations 

--UNION (Use union all to retain duplicates): 
--1. Find courses that ran in Fall 2009 or in Spring 2010 
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
union all
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;

--INTERSECT (Use intersect all to retain duplicates): 
--2. Find courses that ran in Fall 2009 and in spring 2010 
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
intersect all
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;

--MINUS: 
--3. Find courses that ran in Fall 2009 but not in Spring 2010 
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
minus
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;

--Null values 
--4. Find the name of the course for which none of the students registered. 
select course.title
from course
where course.course_id not in (select course_id from takes);

--Nested Subqueries 

--Set Membership (in / not in): 
--5. Find courses offered in Fall 2009 and in Spring 2010. 
select distinct course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
    and course.course_id in (select course_id
                            from takes
                            where semester='Spring' and year=2010);

--6. Find the total number of students who have taken course taught by the instructor with ID 10101.
select Count(student.ID)
from student,takes
where student.ID = takes.ID and takes.course_id in (select course_id
                                                    from teaches
                                                    where ID=10101);

--7. Find courses offered in Fall 2009 but not in Spring 2010. 
select distinct course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
    and course.course_id not in (select course_id
                                from takes
                                where semester='Spring' and year=2010);

--8. Find the names of all students whose name is same as the instructor’s name.
select name
from student
where name in (select name
                from instructor);