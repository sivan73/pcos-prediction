document.addEventListener("DOMContentLoaded", function () {
    let currentStep = 1;
    const totalSteps = 5;

    // Function to calculate and update BMI
    function updateBMI() {
        const weight = parseFloat(document.getElementById('weight').value);
        const height = parseFloat(document.getElementById('height').value) / 100; // Convert cm to meters
        if (weight && height) {
            const bmi = (weight / (height * height)).toFixed(2);
            document.getElementById('bmi').value = bmi;
        } else {
            document.getElementById('bmi').value = '';
        }
    }

    // Function to navigate between steps
    function showStep(step) {
        for (let i = 1; i <= totalSteps; i++) {
            document.getElementById(`step${i}`).classList.remove('active');
        }
        document.getElementById(`step${step}`).classList.add('active');
    }

    // Event listener for the "Next" button
    const nextButtons = document.querySelectorAll('.next-btn');
    nextButtons.forEach(button => {
        button.addEventListener('click', function () {
            currentStep++;
            if (currentStep <= totalSteps) {
                showStep(currentStep);
            } else {
                // Review the form
                showReview();
            }
        });
    });

    // Event listener for keyboard enter key
    document.querySelectorAll('input, select').forEach(input => {
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                currentStep++;
                if (currentStep <= totalSteps) {
                    showStep(currentStep);
                } else {
                    showReview();
                }
            }
        });
    });

    // Function to show review details
    function showReview() {
        let reviewDetails = document.querySelector('.review-details');
        reviewDetails.innerHTML = `
            <h3>Review your details:</h3>
            <p><strong>Name:</strong> ${document.getElementById('name').value}</p>
            <p><strong>Age:</strong> ${document.getElementById('age').value}</p>
            <p><strong>Weight:</strong> ${document.getElementById('weight').value}</p>
            <p><strong>Height:</strong> ${document.getElementById('height').value}</p>
            <p><strong>BMI:</strong> ${document.getElementById('bmi').value}</p>
            <p><strong>Blood Group:</strong> ${document.getElementById('bloodGroup').value}</p>
            <p><strong>Cycle Regularity:</strong> ${document.getElementById('cycleRegular').value}</p>
            <p><strong>Pregnant:</strong> ${document.getElementById('pregnant').value}</p>
            <p><strong>Number of Abortions:</strong> ${document.getElementById('abortions').value}</p>
            <p><strong>Cycle Length:</strong> ${document.getElementById('cycleLength').value}</p>
            <p><strong>Hair Loss:</strong> ${document.getElementById('hairLoss').value}</p>
            <p><strong>Hair Growth:</strong> ${document.getElementById('hairGrowth').value}</p>
            <p><strong>Skin Darkening:</strong> ${document.getElementById('skinDarkening').value}</p>
            <p><strong>Pimples:</strong> ${document.getElementById('pimples').value}</p>
            <p><strong>Weight Gain:</strong> ${document.getElementById('weightGain').value}</p>
            <p><strong>Pulse Rate:</strong> ${document.getElementById('pulseRate').value}</p>
            <p><strong>Respiratory Rate:</strong> ${document.getElementById('respiratoryRate').value}</p>
            <p><strong>BP Systolic:</strong> ${document.getElementById('bpSystolic').value}</p>
            <p><strong>BP Diastolic:</strong> ${document.getElementById('bpDiastolic').value}</p>
            <p><strong>Hemoglobin:</strong> ${document.getElementById('hemoglobin').value}</p>
            <p><strong>FSH:</strong> ${document.getElementById('fsh').value}</p>
            <p><strong>Waist:</strong> ${document.getElementById('waist').value}</p>
            <p><strong>Hip:</strong> ${document.getElementById('hip').value}</p>
        `;
        showStep('review');
    }

    // Event listeners for input changes to update BMI
    document.getElementById('weight').addEventListener('input', updateBMI);
    document.getElementById('height').addEventListener('input', updateBMI);
});