// Button interaction
document.getElementById('clickme').addEventListener('click', function() {
    alert('Thank you for clicking! Stay tuned for more content.');
});

// Form validation
document.getElementById('contact-form').addEventListener('submit', function(event) {
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const message = document.getElementById('message').value;

    if (name === '' || email === '' || message === '') {
        event.preventDefault();
        alert('Please fill out all fields before submitting.');
    } else {
        alert('Your message has been submitted successfully!');
    }
});
