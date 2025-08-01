// Check if an alert message is provided and display it
var chartStudent = $.parseJSON('{{ chartStudent | tojson | safe }}'); // Injected from Flask
var chartGrade = $.parseJSON('{{ chartGrade | tojson | safe }}'); // Injected from Flask
console.log(chartStudent);
console.log(chartGrade);

$(document).ready(function () {
	createLineChart(chartStudent, chartGrade, 'lineChart');
});



let createLineChart = function (chartLabel, chartData, chartElement) {
	// Example chart data for a line chart
	var chartData = {
		type: 'line',
		data: {
			labels: chartLabel,
			datasets: [{
				label:'',
				data: chartData,
				fill: false,
				borderColor: 'rgba(75, 192, 192, 1)',
				borderWidth: 2,
				pointRadius: 5,
				pointBackgroundColor: 'rgba(75, 192, 192, 1)',
				pointBorderColor: 'rgba(75, 192, 192, 1)',
				pointHoverRadius: 7,
				pointHoverBackgroundColor: 'rgba(75, 192, 192, 1)',
				pointHoverBorderColor: 'rgba(75, 192, 192, 1)'
			}]
		}
	};
	// Get the canvas element
	var ctx = document.getElementById(chartElement).getContext('2d');
	// Initialize Chart.js line chart
	var lineChart = new Chart(ctx, chartData);