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
select distinct student.name,course.title,section.year
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

--18. Find the names of all instructors whose department name includes the substring ‘ch’.   
select name
from instructor
where name like '%e%';

--19. List the student names along with the length of the student names. 
select name,length(name) as name_length
from student;

