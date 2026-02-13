$(document).ready(function() {
    $("#submit").on("click", function() {
        let input = $("#numbers").val();
        let numbers = input.split(/\s+/).map(num);
        let divisibleByThree = numbers.filter(num => !isNaN(num) && num % 3 == 0);
        $("#output").text("Numbers divisible by 3: " + (divisibleByThree.length ? divisibleByThree.join(",") : "None"));
    });
});