# University Database Schema

```sql
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
```

## Implement the Following Queries on UNIVERSITY Database

### Set Operations

#### UNION (Use union all to retain duplicates)

**1. Find courses that ran in Fall 2009 or in Spring 2010**

```sql
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
union all
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;
```

**OR**

```sql
select course_id
from section
where semester='Fall' and year=2009
union all
select course_id
from section
where semester='Spring' and year=2010;
```

#### INTERSECT (Use intersect all to retain duplicates)

**2. Find courses that ran in Fall 2009 and in Spring 2010**

```sql
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
intersect all
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;
```

#### MINUS

**3. Find courses that ran in Fall 2009 but not in Spring 2010**

```sql
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
minus
select course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Spring' and year=2010;
```

### Null Values

**4. Find the name of the course for which none of the students registered.**

```sql
select course.title
from course
where course.course_id not in (select course_id from takes);
```

### Nested Subqueries

#### Set Membership (in / not in)

**5. Find courses offered in Fall 2009 and in Spring 2010.**

```sql
select distinct course_id
from takes
where semester='Fall' and year=2009
    and course_id in (select course_id
                        from takes
                        where semester='Spring' and year=2010);
```

**6. Find the total number of students who have taken course taught by the instructor with ID 10101.**

```sql
select count(distinct student.ID)
from student,takes
where student.ID = takes.ID and takes.course_id in (select course_id
                                                    from teaches
                                                    where ID=10101);
```

**7. Find courses offered in Fall 2009 but not in Spring 2010.**

```sql
select distinct course.course_id
from course,takes
where course.course_id = takes.course_id and semester='Fall' and year=2009
    and course.course_id not in (select course_id
                                from takes
                                where semester='Spring' and year=2010);
```

**8. Find the names of all students whose name is same as the instructor’s name.**

```sql
select name
from student
where name in (select name
                from instructor);
```

**9. Find names of instructors with salary greater than that of some (at least one) instructor in the Biology department.**

```sql
select name
from instructor
where salary > some(select salary
                    from instructor
                    where dept_name='Biology');
```

**10. Find the names of all instructors whose salary is greater than the salary of all instructors in the Biology department.**

```sql
select name
from instructor
where salary > all(select salary
                    from instructor
                    where dept_name='Biology');
```

**11. Find the departments that have the highest average salary.**

```sql
select dept_name
from instructor
group by dept_name
having avg(salary) in (select max(avg(salary))
                        from instructor
                        group by dept_name);
```

**12. Find the names of those departments whose budget is lesser than the average salary of all instructors.**

```sql
select dept_name
from department
where budget < (select avg(salary)
                    from instructor);
```

### Test for Empty Relations (exists/ not exists)

**13. Find all courses taught in both the Fall 2009 semester and in the Spring 2010 semester.**

```sql
select distinct c.course_id
from course c,takes
where c.course_id = takes.course_id and semester='Fall' and year=2009
and exists(select *
            from course s,takes
            where c.course_id=s.course_id and s.course_id = takes.course_id and semester='Spring' and year=2010);
```

<span style="color:red">**14. Find all students who have taken all courses offered in the Biology department.**</span>

```sql
select s.id
from student s
where not exists((select course_id
                from course
                where dept_name='Biology')
                minus
                (select t.course_id
                from takes t
                where s.id=t.id));
```

### Test for Absence of Duplicate Tuples

<span style="color:red">** **15. Find all courses that were offered at most once in 2009.**</span>

```sql
select course_id
from teaches
where year=2009
group by course_id
having count(*)<=1;
```

<span style="color:red">** **16. Find all the students who have opted at least two courses offered by CSE department.**</span>

```sql
select distinct c.course_id
from course c
where 1>= (select count(s2.course_id)
            from section s2
            where c.course_id=s2.course_id and s2.year=2009);
```

<span style="color:blue">* **17. Find the average instructors salary of those departments where the average salary is greater than 42000.**</span>

```sql
select dept_name,avg(salary)
from instructor
group by dept_name
having avg(salary)>42000;
```

### Views

**18. Create a view all_courses consisting of course sections offered by Physics department in the Fall 2009, with the building and room number of each section.**

```sql
create view all_courses as
select course_id,sec_id,building,room_number
from section
where course_id in (select course_id
                    from course
                    where dept_name='Physics')
    and semester='Fall' and year=2009;
```

**See the view:**

```sql
select * from all_courses;
```

**What is a view?**

A view is a virtual table based on the result-set of an SQL statement. A view contains rows and columns, just like a real table. The fields in a view are fields from one or more real tables in the database. You can add SQL functions, WHERE, and JOIN statements to a view and present the data as if the data were coming from one single table.

**Can we do operations on a view?**

Yes, we can do operations on views. We can create a view from one or more tables and use it as a table in the database. Operations like SELECT, INSERT, UPDATE, and DELETE are possible on views.

**19. Select all the courses from all_courses view.**

```sql
select * from all_courses;
```

**20. Create a view department_total_salary consisting of department name and total salary of that department.**

```sql
create view department_total_salary as
select dept_name,sum(salary) as total_salary
from instructor
group by dept_name;
```

**How to see all views/tables in SQL Plus**

To see all the tables in the database:

```sql
SELECT table_name FROM user_tables;
```

To see all the views in the database:

```sql
select view_name from user_views;
```

### Additional Exercise

**1. Find the names of all departments with instructor and remove duplicates.**

```sql
select distinct dept_name,name
from instructor;
```

**2. For all instructors who have taught some course, find their names and the course ID of the courses they taught.**

```sql
select instructor.name,course_id
from instructor,teaches
where instructor.id = teaches.id;
```

**3. Find all the instructors with the courses they taught.**

```sql
select instructor.name,title
from instructor,teaches,course
where instructor.id = teaches.id and course.course_id = teaches.course_id;
```

<span style="color:blue">* **4. List all the students with student name, department name, advisor name, and the number of courses registered.**</span>

```sql
select s.name,s.dept_name,i.name,count(t.course_id)
from student s,advisor a,instructor i,takes t
where s.id=a.s_id and a.i_id=i.id and s.id=t.id
group by s.name,s.dept_name,i.name;
```
