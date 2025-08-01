const wrapper = document.querySelector('.wrapper');
const btnPopup = document.querySelector('.btnlogin-popup');
const iconClose = document.querySelector('.icon-close');
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const showRegisterFormLink = document.getElementById('showRegisterForm');
const showLoginFormLink = document.getElementById('showLoginForm');

// Function to handle opening the terms modal
function openModal() {
    const modal = document.getElementById('termsModal');
    modal.style.display = 'flex'; // Use flex to center the content vertically
    document.body.classList.add('modal-open');
}

// Function to handle closing the terms modal
function closeModal() {
    const modal = document.getElementById('termsModal');
    modal.style.display = 'none';
    document.body.classList.remove('modal-open');
}

// Function to handle closing the modal without agreement
function closeModalWithoutAgreement() {
    const termsCheckbox = document.getElementById('termsCheckbox');
    termsCheckbox.checked = false;
    closeModal();
}

// Function to handle agreement and close modal
function agreeTerms() {
    const termsCheckbox = document.getElementById('termsCheckbox');
    termsCheckbox.checked = true;
    closeModal();
}

// Function to handle closing the modal without agreement
function closeModalWithoutAgreement() {
    const termsCheckbox = document.getElementById('termsCheckbox');
    termsCheckbox.checked = false;
    closeModal();
}

// Add an event listener to open the terms modal when the checkbox is clicked
const termsCheckbox = document.getElementById('termsCheckbox');
termsCheckbox.addEventListener('click', openModal);

// Update the event listener for the close button in the terms modal
const closeButton = document.querySelector('.modal .close');
closeButton.addEventListener('click', closeModalWithoutAgreement);

// Update the event listener for the "Agree" button in the terms modal
const agreeButton = document.querySelector('.modal button');
agreeButton.addEventListener('click', agreeTerms);

btnPopup.addEventListener('click', () => {
    wrapper.classList.add('active-popup');
    loginForm.style.display = 'block';
    registerForm.style.display = 'none';
});

iconClose.addEventListener('click', () => {
    wrapper.classList.remove('active-popup');
    loginForm.style.display = 'none';
    registerForm.style.display = 'none';
});

showRegisterFormLink.addEventListener('click', () => {
    loginForm.style.display = 'none';
    registerForm.style.display = 'block';
});

showLoginFormLink.addEventListener('click', () => {
    loginForm.style.display = 'block';
    registerForm.style.display = 'none';
});

// Add the following code to check if the terms checkbox is checked before submitting the form
const registerFormElement = document.querySelector('#registerForm form');
registerFormElement.addEventListener('submit', function (event) {
    const termsCheckbox = document.getElementById('termsCheckbox');
    if (!termsCheckbox.checked) {
        alert('Please agree to the terms and conditions before registering.');
        event.preventDefault(); // Prevent the form from submitting
    }
});