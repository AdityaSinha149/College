let editTarget = null;

function addRow() {
  const name = document.getElementById("empName").value.trim();
  const pos = document.getElementById("empPos").value.trim();
  const salary = document.getElementById("empSalary").value.trim();

  if (!name || !pos || !salary) {
    alert("All fields are required.");
    return;
  }

  const table = document.getElementById("empTable").querySelector("tbody");
  const rowCount = table.rows.length + 1;

  const row = table.insertRow();

  const cellIndex = row.insertCell(0);
  const cellName = row.insertCell(1);
  const cellPos = row.insertCell(2);
  const cellSalary = row.insertCell(3);
  const cellActions = row.insertCell(4);

  cellIndex.innerHTML = rowCount;
  cellName.innerHTML = name;
  cellPos.innerHTML = pos;
  cellSalary.innerHTML = salary;
  cellActions.innerHTML = `
    <button class="btn btn-sm btn-warning" onclick="editRow(this)">Edit</button>
    <button class="btn btn-sm btn-danger" onclick="deleteRow(this)">Delete</button>
  `;

  clearInputs();
}

function editRow(btn) {
  const row = btn.closest("tr");

  if (editTarget === row) return;

  editTarget = row;

  const name = row.cells[1].innerText;
  const pos = row.cells[2].innerText;
  const salary = row.cells[3].innerText;

  document.getElementById("empName").value = name;
  document.getElementById("empPos").value = pos;
  document.getElementById("empSalary").value = salary;

  const addBtn = document.querySelector("button[onclick='addRow()']");
  addBtn.innerText = "Update";
  addBtn.classList.replace("btn-primary", "btn-success");
  addBtn.setAttribute("onclick", "updateRow()");
}

function updateRow() {
  if (!editTarget) return;

  const name = document.getElementById("empName").value.trim();
  const pos = document.getElementById("empPos").value.trim();
  const salary = document.getElementById("empSalary").value.trim();

  if (!name || !pos || !salary) {
    alert("All fields are required.");
    return;
  }

  editTarget.cells[1].innerText = name;
  editTarget.cells[2].innerText = pos;
  editTarget.cells[3].innerText = salary;

  resetForm();
}

function deleteRow(btn) {
  const row = btn.closest("tr");
  row.remove();
  renumberRows();
}

function renumberRows() {
  const rows = document.getElementById("empTable").querySelector("tbody").rows;

  for (let i = 0; i < rows.length; i++) {
    rows[i].cells[0].innerText = i + 1;
  }
}

function clearInputs() {
  document.getElementById("empName").value = "";
  document.getElementById("empPos").value = "";
  document.getElementById("empSalary").value = "";
}

function resetForm() {
  clearInputs();

  editTarget = null;

  const addBtn = document.querySelector("button[onclick='updateRow()']");
  addBtn.innerText = "Add";
  addBtn.classList.replace("btn-success", "btn-primary");
  addBtn.setAttribute("onclick", "addRow()");
}
