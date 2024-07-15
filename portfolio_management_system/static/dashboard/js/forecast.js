document.getElementById('forecast-report-btn').addEventListener('click', function () {
    $('#forecastReportModal').modal('show');
});

document.getElementById('forecast-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    const stockCode = document.getElementById('stock-code').value;
    const startDate = document.getElementById('start-date').value;
    const predictionDays = document.getElementById('prediction-days').value;
    console.log("test");
    
    // Call your backend API to get forecast data
    fetch('http://127.0.0.1:8000/lstm_stock_prediction/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ stockCode, startDate, predictionDays })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        if (data.success) {
            document.getElementById('forecast-results').style.display = 'block';

            // Render LSTM chart
            new Chart(document.getElementById('forecast-chart'), {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [{
                        label: `${data.model} Forecast`,
                        data: data.predictions,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2
                    }]
                },
                options: { responsive: true }
            });
            
            document.getElementById('mse').textContent = `MSE: ${data.mse}`;
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Error fetching data. Please try again later.');
    });
});

