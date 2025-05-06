document.getElementById("signupForm").addEventListener("submit", function(e) {
  e.preventDefault();
  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;

  alert(`Welcome ${name}, your email is ${email}`);
  // You can send data to Python backend here using fetch()
});
