# Lab 1 — HTML and CSS Basics (Additional Questions)

**Name:** Aditya Sinha<br>
**Reg. No:** 230905218<br>
**Class & Section:** CSE-A1<br>
**Roll No:** 27

---

## Additional Question 1

1. Create the following output in HTML

**Code:**

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Q1 Population Table</title>
  </head>
  <body>
    <h3>Population (in Crores)</h3>

    <table border="1" cellpadding="8">
      <tr bgcolor="red">
        <th>Country</th>
        <th>Year</th>
        <th>Population</th>
      </tr>

      <tr>
        <td bgcolor="lightyellow">India</td>
        <td bgcolor="lightblue">1998</td>
        <td bgcolor="lightgreen">85</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">India</td>
        <td bgcolor="lightblue">1999</td>
        <td bgcolor="lightgreen">90</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">India</td>
        <td bgcolor="lightblue">2000</td>
        <td bgcolor="lightgreen">100</td>
      </tr>

      <tr>
        <td bgcolor="lightyellow">USA</td>
        <td bgcolor="lightblue">1998</td>
        <td bgcolor="lightgreen">30</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">USA</td>
        <td bgcolor="lightblue">1999</td>
        <td bgcolor="lightgreen">35</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">USA</td>
        <td bgcolor="lightblue">2000</td>
        <td bgcolor="lightgreen">40</td>
      </tr>

      <tr>
        <td bgcolor="lightyellow">UK</td>
        <td bgcolor="lightblue">1998</td>
        <td bgcolor="lightgreen">25</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">UK</td>
        <td bgcolor="lightblue">1999</td>
        <td bgcolor="lightgreen">30</td>
      </tr>
      <tr>
        <td bgcolor="lightyellow">UK</td>
        <td bgcolor="lightblue">2000</td>
        <td bgcolor="lightgreen">35</td>
      </tr>
    </table>
  </body>
</html>
```

**Output:**
![](pics/a1.png)

---

## Additional Question 2

2. Create an array of JavaScript objects for the data in question 1 and display each row in the table form through JavaScript code

**Code:**

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Population Table</title>
  </head>
  <body>
    <h3>Population (in Crores)</h3>

    <table border="1" cellpadding="8">
      <thead bgcolor="red">
        <tr>
          <th>Country</th>
          <th>Year</th>
          <th>Population</th>
        </tr>
      </thead>

      <tbody id="tableBody">
        <!-- Rows from JavaScript -->
      </tbody>
    </table>

    <script>
      var data = [
        { country: "India", year: 1998, population: 85 },
        { country: "India", year: 1999, population: 90 },
        { country: "India", year: 2000, population: 100 },

        { country: "USA", year: 1998, population: 30 },
        { country: "USA", year: 1999, population: 35 },
        { country: "USA", year: 2000, population: 40 },

        { country: "UK", year: 1998, population: 25 },
        { country: "UK", year: 1999, population: 30 },
        { country: "UK", year: 2000, population: 35 },
      ];

      var table = document.getElementById("tableBody");

      for (var i = 0; i < data.length; i++) {
        var row =
          "<tr>" +
          "<td bgcolor='lightyellow'>" +
          data[i].country +
          "</td>" +
          "<td bgcolor='lightblue'>" +
          data[i].year +
          "</td>" +
          "<td bgcolor='lightgreen'>" +
          data[i].population +
          "</td>" +
          "</tr>";

        table.innerHTML += row;
      }
    </script>
  </body>
</html>
```

**Output:**
![](pics/a2.png)
