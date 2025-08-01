document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('btnn').addEventListener('click', goToAnalysis);

    let timerInterval;
    let totalSeconds = 0;
    let captureEnded = false;

    // Initially disable the Analysis button
    document.getElementById('btnn').disabled = true;

    function updateButtonState() {
        const btnn = document.getElementById('btnn');
        btnn.classList.toggle('active', !btnn.disabled);
    }

    updateButtonState();

    document.getElementById('btn').addEventListener('click', function () {
        document.getElementById('btn').style.display = 'none';
        document.getElementById('btnnn').style.display = 'inline-block';
        captureEnded = false;

        // Enable Analysis button when starting capture
        document.getElementById('btnn').disabled = false;

        // Start the timer or resume from the last recorded time
        timerInterval = setInterval(function () {
            totalSeconds++;
            updateDurationDisplay();
        }, 1000);
    });

    document.getElementById('btnnn').addEventListener('click', function () {
        document.getElementById('btn').style.display = 'inline-block';
        document.getElementById('btnnn').style.display = 'none';
        captureEnded = true;

        // Stop the timer
        clearInterval(timerInterval);
        updateDurationDisplay(); // Update display with the final time

        // Enable Analysis button after End Capture
        document.getElementById('btnn').disabled = false;
        document.getElementById('btnn').classList.remove('active'); // Ensure the active class is removed
        document.getElementById('btnn').style.backgroundColor = '#2382BC'; // Colored
        document.getElementById('btnn').style.cursor = 'pointer'; // Hand cursor

        // Remove the beforeunload event listener when the capture ends
        window.onbeforeunload = null;
    });

    function updateDurationDisplay() {
        let minutes = Math.floor(totalSeconds / 60);
        let seconds = totalSeconds % 60;
        document.getElementById('duration').innerText = `${minutes} minutes ${seconds} seconds`;

        const btnn = document.getElementById('btnn');

        // Disable Analysis button during capture
        if (!captureEnded) {
            btnn.disabled = true;
            btnn.classList.remove('active');
            btnn.style.backgroundColor = '#545454'; // Gray color
            btnn.style.cursor = 'default'; // Default cursor
        } else {
            // Enable Analysis button after capture ends
            btnn.disabled = false;
            btnn.classList.add('active');
            btnn.style.backgroundColor = '#2382BC'; // Colored
            btnn.style.cursor = 'pointer'; // Hand cursor
        }
    }

    document.getElementById('btn').addEventListener('click', function () {
        totalSeconds = 0; // Reset the timer
        updateDurationDisplay(); // Update display with the reset time

        // Disable Analysis button when starting a new capture
        document.getElementById('btnn').disabled = true;
        document.getElementById('btnn').classList.remove('active'); // Ensure the active class is removed
        document.getElementById('btnn').style.backgroundColor = '#545454'; // Gray color
        document.getElementById('btnn').style.cursor = 'default'; // Default cursor

        // Remove the beforeunload event listener when starting a new capture
        window.onbeforeunload = null;
    });

    function goToAnalysis() {
        if (captureEnded) {
            window.location.href = "analysis.html";
        } else {
            // Check if the capture has started
            if (totalSeconds > 0) {
                var confirmation = confirm("The capture is still ongoing. Do you want to proceed to analysis anyway?");

                // Whether the user clicks "OK" or "Cancel", do not redirect and do not stop the timer
                alert("Analysis canceled. The capture is still ongoing.");
                // Add additional logic here if needed

                // The timer will continue running, and the user will stay on the page
            } else {
                // The capture has not started, redirect to analysis without showing the confirmation
                window.location.href = "analysis.html";
            }
        }
    }

    // Add an event listener to the "Detect" navigation link
    document.getElementById('btnDetect').addEventListener('click', function () {
        // Disable Analysis button when "Detect" link is clicked
        document.getElementById('btnn').disabled = true;
        document.getElementById('btnn').classList.remove('active');
        document.getElementById('btnn').style.backgroundColor = '#545454'; // Gray color
        document.getElementById('btnn').style.cursor = 'default'; // Default cursor
    });

    // script.js
    document.getElementById('btn').addEventListener('click', function() {
        // Assuming startCapture function is defined in script.js or loaded before this line
        startCapture();
        // Add your other code here
    });

// script1.js

    function getRaspberryPiBaseUrl() {
        // Dynamically determine the base URL of the Raspberry Pi API
        const protocol = window.location.protocol;
        const host = window.location.hostname;
        const port = window.location.port || "5001";  // Use port 80 if not specified

        return `${protocol}//${host}:${port}`;
    }

    function startCapture() {
        const baseUrl = getRaspberryPiBaseUrl();

        fetch(`${baseUrl}/capture`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'action=start',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // Handle success, update UI, etc.
        })
        .catch(error => {
            console.error(error);
            // Handle error, show alert, etc.
        });
    }

    function stopCapture() {
        const baseUrl = getRaspberryPiBaseUrl();

        fetch(`${baseUrl}/capture`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'action=stop',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // Handle success, update UI, etc.
        })
        .catch(error => {
            console.error(error);
            // Handle error, show alert, etc.
        });
	}

    const url = 'http://192.168.104.171:5001/capture';
    const data = new FormData();
    data.append('action', 'start');

    fetch(url, {
        method: 'POST',
        body: data,
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        // Handle success, if needed
    })
    .catch((error) => {
        console.error('Error:', error);
        // Handle error, if needed
	});
});
