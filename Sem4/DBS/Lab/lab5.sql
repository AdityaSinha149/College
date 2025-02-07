/*
employee1(fname,minit,,lname,ssn,bdate,address,sex,salary,super_ssn,dno)
department1(dname,dnumber,mgr_ssn,mgr_start_date)
dept_locations(dnumber,dlocation)
project(pname,pnumber,plocation,dnum)
works_on(essn,pno)
dependent(essn,dependent_name,sex,bdate,relationship)
*/

--1. Retrieve the birth date and address of the employee(s) whose name is ‘John B. 
--Smith’. Retrieve the name and address of all employees who work for the 
--‘Research’ department.
select bdate,address
from employee1
where fname='John' and minit='B' and lname='Smith';

select fname,minit,lname,address
from employee1,department
where dno=dnumber and dname='Research';

--2. For every project located in ‘Stanford’, list the project number, the controlling  
--department number, and the department manager’s last name, address, and birth date.
select pnumber,p.dnum,lname,address,bdate
from employee1 e,department1 d,project p
where p.dnum=d.dnumber and mgr_ssn=ssn and plocation='Stafford';

--3. For each employee, retrieve the employee’s first and last name and the first and 
--last name of his or her immediate supervisor.
select e1.fname,e1.lname,e2.fname,e2.lname
from employee1 e1,employee1 e2
where e1.super_ssn=e2.ssn;

--4. Make a list of all project numbers for projects that involve an employee whose 
--last name is ‘Smith’, either as a worker or as a manager of the department that 
--controls the project.
select pnumber
from project,department1,employee1
where dnum=dnumber and mgr_ssn=ssn and lname='Smith'
union
select pno
from works_on,employee1
where essn=ssn and lname='Smith';

--5. Show the resulting salaries if every employee working on the ‘ProductX’ project 
--is given a 10 percent raise.
select salary*1.1
from employee1,works_on,project
where pname='ProductX' and pnumber=pno and essn=ssn;

--6. Retrieve a list of employees and the projects they are working on, ordered by 
--department and, within each department, ordered alphabetically by last name, then 
--first name.
select
from employee1,works_on,project
where pnumber=pno and essn=ssn
order by

--13. 
with more_than_5(dno) as(
    select dno
    from employee1
    group by dno
    having count(dno)>5
)
