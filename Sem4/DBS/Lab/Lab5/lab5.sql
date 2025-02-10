/*
employee(fname,minit,,lname,ssn,bdate,address,sex,salary,super_ssn,dno)
department(dname,dnumber,mgr_ssn,mgr_start_date)
dept_locations(dnumber,dlocation)
project(pname,pnumber,plocation,dnum)
works_on(essn,pno)
dependent(essn,dependent_name,sex,bdate,relationship)
*/

--1. Retrieve the birth date and address of the employee(s) whose name is ‘John B. 
--Smith’. Retrieve the name and address of all employees who work for the 
--‘Research’ department.
select bdate,address
from employee
where fname='John' and minit='B' and lname='Smith';

select fname,minit,lname,address
from employee,department
where dno=dnumber and dname='Research';

--2. For every project located in ‘Stanford’, list the project number, the controlling  
--department number, and the department manager’s last name, address, and birth date.
select pnumber,p.dnum,lname,address,bdate
from employee e,department d,project p
where p.dnum=d.dnumber and mgr_ssn=ssn and plocation='Stafford';

--3. For each employee, retrieve the employee’s first and last name and the first and 
--last name of his or her immediate supervisor.
select e1.fname,e1.lname,e2.fname,e2.lname
from employee e1,employee e2
where e1.super_ssn=e2.ssn;

--4. Make a list of all project numbers for projects that involve an employee whose 
--last name is ‘Smith’, either as a worker or as a manager of the department that 
--controls the project.
select pnumber
from project,department,employee
where dnum=dnumber and mgr_ssn=ssn and lname='Smith'
union
select pno
from works_on,employee
where essn=ssn and lname='Smith';

--5. Show the resulting salaries if every employee working on the ‘ProductX’ project 
--is given a 10 percent raise.
select salary*1.1
from employee,works_on,project
where pname='ProductX' and pnumber=pno and essn=ssn;

--6. Retrieve a list of employees and the projects they are working on, ordered by 
--department and, within each department, ordered alphabetically by last name, then 
--first name.
select dname,lname,fname,pname
from employee,works_on,project,department
where pnumber=pno and essn=ssn and dno=dnumber
order by dname,lname,fname;

--7. 
select fname,minit,lname
from employee e,dependent d
where ssn=essn and fname=dependent_name and e.sex=d.sex;

--8. Retrieve the names of employees who have no dependents. 
SELECT e.fname || ' ' || e.minit || ' ' || e.lname AS employee_name
FROM employee e
WHERE e.ssn NOT IN (SELECT d.essn FROM dependent d);

--Not 8 but better. Retrieve the names of employees who are not supervisor to anyone. 
with supervisor(ssn) as(
    select distinct ssn
    from employee
    where ssn in(
        select super_ssn
        from employee
    )
)
select fname || ' ' || minit || ' ' || lname as name
from employee
where ssn not in (
    select ssn
    from supervisor
);

--OR

with supervisor(ssn) as(
    select distinct super_ssn as ssn
    from employee
    WHERE super_ssn IS not null  -- Exclude NULLs
)
select fname || ' ' || minit || ' ' || lname as name
from employee
where ssn not in (
    select ssn
    from supervisor
);

--But this is wrong -

with supervisor(ssn) as (
    select distinct super_ssn as ssn
    from employee
)
select fname || ' ' || minit || ' ' || lname as name
from employee
WHERE ssn not in (
    select ssn from supervisor
);

--SQL gives false if u check for not in a set and the set has null values.

/*
Example 1: in with NULL
sql
Copy
Edit
select * from employee WHERE ssn in (123, 456, NULL);
If ssn = 123 → ✅ TRUE (matches 123)
If ssn = 456 → ✅ TRUE (matches 456)
If ssn = 789 (not in the set, but set contains NULL) → UNKNOWN (treated as FALSE)
→ No rows are returned for 789.
Example 2: not in with NULL (Dangerous Case)
sql
Copy
Edit
select * from employee WHERE ssn not in (123, 456, NULL);
If ssn = 123 → ❌ FALSE (excluded)
If ssn = 456 → ❌ FALSE (excluded)
If ssn = 789 → ⚠️ UNKNOWN (because NULL exists in the set)
→ Since not in must be fully TRUE for a row to be included, it excludes all rows!
*/

--9. List the names of managers who have at least one dependent.
SELECT DISTINCT e.fname || ' ' || e.minit || ' ' || e.lname AS manager_name
FROM employee e JOIN department d
ON e.ssn = d.mgr_ssn
WHERE e.ssn IN (SELECT essn FROM dependent);


--10. Find the sum of the salaries of all employees, the maximum salary, the minimum 
--salary, and the average salary. 
with salSum as (
    select sum(salary) as total_salary from employee
),
maxSal as (
    select max(salary) as max_salary from employee
),
minSal as (
    select min(salary) as min_salary from employee
),
avgSal as (
    select avg(salary) as avg_salary from employee
)
select * 
from salSum, maxSal, minSal, avgSal;

--11. For each project, retrieve the project number, the project name, and the number 
--of employees who work on that project. 
select pnumber,pname,count(essn)
from project,works_on
where pnumber=pno
group by pnumber,pname;

--12. For each project on which more than two employees work, retrieve the project 
--number, the project name, and the number of employees who work on the 
--project. 
select pnumber,pname,count(essn)
from project,works_on
where pnumber=pno
group by pnumber,pname
having count(essn)>2;

--13. For each department that has more than five employees, retrieve the department 
--number and the number of its employees who are making more than 40,000. 
with more_than_5(dno) as(
    select dno
    from employee
    group by dno
    having count(dno)>5
)
select employee.dno,count(ssn)
from employee,more_than_5
where employee.dno=more_than_5.dno and salary>40000
group by employee.dno;
