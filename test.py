Download Node.js and install
Go to Windows Powershell Run as Administrator type Set-ExecutionPolicy RemoteSigned
Open VS Code, terminal and then proceed further
1. Scientific Calculator
Open Terminal in VS Code
Step 1. npx create-react-app scientific-calculator
Step 2. cd scientific-calculator
Step 3. npm start
App.js
import React, { useState } from "react";
import "./App.css";
const buttons = [
 "C", "(", ")", "←",
 "sin", "cos", "tan", "√",
 "7", "8", "9", "/",
 "4", "5", "6", "*",
 "1", "2", "3", "-",
 "0", ".", "=", "+"
];
export default function App() {
 const [input, setInput] = useState("");
 const handleClick = (value) => {
 if (value === "C") return setInput("");
 if (value === "←") return setInput(input.slice(0, -1));
 if (value === "=") {
 try {
 // Evaluate safe expressions only (simple eval)
 const result = eval(
 input
 .replace(/√/g, "Math.sqrt")
 .replace(/sin/g, "Math.sin")
 .replace(/cos/g, "Math.cos")
 .replace(/tan/g, "Math.tan")
 );
 return setInput(result.toString());
 } catch {
 return setInput("Error");
 }
 }
 setInput(input + value);
 };
 return (
 <div className="calculator">
 <div className="display">{input || "0"}</div>
 <div className="buttons">
 {buttons.map((btn, i) => (
 <button key={i} onClick={() => handleClick(btn)}>
 {btn}
 </button>
 ))}
 </div>
 </div>
 );
}
App.css
body {
 margin: 0;
 padding: 0;
 height: 100vh;
 display: flex;
 align-items: center;
 justify-content: center;
 background: #f0f0f0;
 font-family: sans-serif;
}
.calculator {
 background: #fff;
 padding: 20px;
 border-radius: 10px;
 box-shadow: 0 5px 15px rgba(0,0,0,0.1);
 width: 300px;
}
.display {
 background: #eee;
 color: #333;
 font-size: 1.8rem;
 padding: 15px;
 border-radius: 5px;
 margin-bottom: 10px;
 text-align: right;
 height: 50px;
 overflow-x: auto;
}
.buttons {
 display: grid;
 grid-template-columns: repeat(4, 1fr);
 gap: 8px;
}
button {
 padding: 12px;
 font-size: 1rem;
 background: #ddd;
 border: none;
 border-radius: 5px;
 cursor: pointer;
}
button:hover {
 background: #ccc;
}
2. Compass Clock
Open Terminal in VS Code
Step 1. npx create-react-app compass-clock
Step 2. cd compass-clock
Step 3. npm start
App.js
import React, { useState, useEffect } from "react";
function CompassClock() {
 const [time, setTime] = useState(new Date());
 useEffect(() => {
 const timer = setInterval(() => setTime(new Date()), 1000);
 return () => clearInterval(timer);
 }, []);
 const hours = time.getHours() % 12;
 const minutes = time.getMinutes();
 const seconds = time.getSeconds();
 // Calculate rotation angles for hands (like clock)
 // Map to compass directions: 12 o'clock = North, 3 o'clock = East, etc.
 const hourAngle = (hours + minutes / 60) * 30; // 360/12 = 30deg per hour
 const minuteAngle = minutes * 6; // 360/60 = 6deg per minute
 const secondAngle = seconds * 6; // 360/60 = 6deg per second
 // Format date string
 const dateString = time.toLocaleDateString(undefined, {
 weekday: "short",
 year: "numeric",
 month: "short",
 day: "numeric",
 });
 return (
 <div style={{ width: 300, height: 300, margin: "auto", userSelect: "none" }}>
 <svg viewBox="0 0 200 200" width="100%" height="100%">
 {/* Outer compass circle */}
 <circle cx="100" cy="100" r="90" stroke="#333" strokeWidth="4" fill="#f8f8f8" />
 {/* Compass directions */}
 <text x="100" y="30" textAnchor="middle" fontWeight="bold" fontSize="16">N</text>
 <text x="170" y="105" textAnchor="middle" fontWeight="bold" fontSize="16">E</text>
 <text x="100" y="180" textAnchor="middle" fontWeight="bold" fontSize="16">S</text>
 <text x="30" y="105" textAnchor="middle" fontWeight="bold" fontSize="16">W</text>
 {/* Hour hand */}
 <line
 x1="100"
 y1="100"
 x2={100 + 40 * Math.sin((Math.PI / 180) * hourAngle)}
 y2={100 - 40 * Math.cos((Math.PI / 180) * hourAngle)}
 stroke="#333"
 strokeWidth="6"
 strokeLinecap="round"
 />
 {/* Minute hand */}
 <line
 x1="100"
 y1="100"
 x2={100 + 60 * Math.sin((Math.PI / 180) * minuteAngle)}
 y2={100 - 60 * Math.cos((Math.PI / 180) * minuteAngle)}
 stroke="#666"
 strokeWidth="4"
 strokeLinecap="round"
 />
 {/* Second hand */}
 <line
 x1="100"
 y1="100"
 x2={100 + 70 * Math.sin((Math.PI / 180) * secondAngle)}
 y2={100 - 70 * Math.cos((Math.PI / 180) * secondAngle)}
 stroke="#ff0000"
 strokeWidth="2"
 strokeLinecap="round"
 />
 {/* Center dot */}
 <circle cx="100" cy="100" r="5" fill="#333" />
 {/* Date display below compass */}
 <text
 x="100"
 y="195"
 textAnchor="middle"
 fontSize="14"
 fill="#333"
 fontFamily="Arial, sans-serif"
 >
 {dateString}
 </text>
 </svg>
 </div>
 );
}
export default CompassClock;
App.css
.App {
 text-align: center;
}
.App-logo {
 height: 40vmin;
 pointer-events: none;
}
@media (prefers-reduced-motion: no-preference) {
 .App-logo {
 animation: App-logo-spin infinite 20s linear;
 }
}
.App-header {
 background-color: #282c34;
 min-height: 100vh;
 display: flex;
 flex-direction: column;
 align-items: center;
 justify-content: center;
 font-size: calc(10px + 2vmin);
 color: white;
}
.App-link {
 color: #61dafb;
}
@keyframes App-logo-spin {
 from {
 transform: rotate(0deg);
 }
 to {
 transform: rotate(360deg);
 }
}
3. Voting App
Open Terminal in VS Code
Step 1. npx create-react-app voting-app
Step 2. cd voting-app
Step 3. npm start
App.js
import React, { useState } from 'react';
import './voting.css';
function App() {
 const [votes, setVotes] = useState({
 candidateA: 0,
 candidateB: 0
 });
 const castVote = (candidate) => {
 setVotes((prevVotes) => ({
 ...prevVotes,
 [candidate]: prevVotes[candidate] + 1
 }));
 };
 return (
 <div>
 <h1>College Voting System</h1>
 <button onClick={() => castVote('candidateA')}>Vote for Candidate A</button>
 <button onClick={() => castVote('candidateB')}>Vote for Candidate B</button>
 <h2>Results:</h2>
 <p>Candidate A: {votes.candidateA} votes</p>
 <p>Candidate B: {votes.candidateB} votes</p>
 </div>
 );
}
export default App;
App.css
body {
 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
 background-color: #f5f7fa;
 margin: 0;
 padding: 20px;
 text-align: center;
}
h1 {
 color: #333;
 margin-bottom: 30px;
}
button {
 background-color: #007bff;
 border: none;
 color: white;
 padding: 12px 24px;
 margin: 10px;
 border-radius: 6px;
 cursor: pointer;
 font-size: 16px;
 transition: background-color 0.3s ease;
}
button:hover {
 background-color: #0056b3;
}
h2 {
 color: #444;
 margin-top: 40px;
}
p {
 font-size: 18px;
 color: #555;
}
4. Quick-Health Reference using Angular
Create a Index.html file and then paste the code
Index.html
<!DOCTYPE html>
<html ng-app="app" ng-controller="c">
<head>
<meta charset="utf-8">
<title>Health Reference</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/angular.js/1.8.3/angular.min.js"></script>
<style>
body{font-family:Arial,sans-serif;background:#f4f6f8;margin:0;padding:40px;display:flex;justify-content:center}
.container{background:#fff;padding:24px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);max-width:600px;width:100%}
h2{font-size:28px;margin:0 0 20px;text-align:center;color:#222}
.tabs{display:flex;gap:10px;margin-bottom:20px}
.tabs button{flex:1;padding:12px;border:none;border-radius:4px;cursor:pointer;font-size:16px;font-weight:bold}
.card{border:1px solid #ddd;padding:14px;border-radius:6px;margin-bottom:10px;background:#fafafa;font-size:16px}
.labs{color:#16a34a}.symptoms{color:#2563eb}
</style>
</head>
<body>
<div class="container">
<h2>Health Reference</h2>
<div class="tabs">
<button ng-click="t='labs'" ng-style="{'background-color':t==='labs'?'#3b82f6':'#e5e7eb','color':t==='labs'?'#fff':'#000'}">Lab
Reports</button>
<button ng-click="t='symptoms'" ng-style="{'backgroundcolor':t==='symptoms'?'#3b82f6':'#e5e7eb','color':t==='symptoms'?'#fff':'#000'}">Symptoms</button>
</div>
<div ng-if="t==='labs'"><div class="card" ng-repeat="l in labs"><b>{{l.test}}</b><div class="labs">Normal:
{{l.range}}</div></div></div>
<div ng-if="t==='symptoms'"><div class="card" ng-repeat="s in symptoms"><b>{{s.symptom}}</b><div
class="symptoms">{{s.action}}</div></div></div>
</div>
<script>
angular.module('app',[]).controller('c',function($scope){
$scope.t='labs';
$scope.labs=[{test:'Blood Sugar',range:'70-100 mg/dL'},{test:'Blood Pressure',range:'120/80 mmHg'},{test:'Cholesterol',range:'<200
mg/dL'}];
$scope.symptoms=[{symptom:'Fever',action:'Rest and hydrate'},{symptom:'Headache',action:'Take pain
relief'},{symptom:'Cough',action:'Drink warm liquids'}];
});
</script>
</body>
</html>
6. Tic-Tac-Toe using Angular
Create a Index.html file and then paste the code
Index.html
<!DOCTYPE html>
<html ng-app="app">
<head>
 <meta charset="UTF-8">
 <title>Tic Tac Toe</title>
 <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/angular-material@1.2.2/angular-material.min.css">
 <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>
 <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular-animate.min.js"></script>
 <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular-aria.min.js"></script>
 <script src="https://cdn.jsdelivr.net/npm/angular-material@1.2.2/angular-material.min.js"></script>
 <style>
 body { font-family: sans-serif; text-align: center; padding: 40px; }
 .grid { display: grid; grid-template-columns: repeat(3, 60px); gap: 6px; justify-content: center; margin: 20px 0; }
 .cell { height: 60px; font-size: 24px; }
 </style>
</head>
<body ng-controller="ctrl">
 <h3>Tic Tac Toe</h3>
 <div ng-if="winner">Winner: {{winner}}</div>
 <div class="grid">
 <md-button class="md-raised cell" ng-repeat="c in board track by $index" ng-click="play($index)">{{c}}</md-button>
 </div>
 <md-button class="md-primary md-raised" ng-click="reset()">Reset</md-button>
 <script>
 angular.module('app', ['ngMaterial']).controller('ctrl', $scope => {
 const win = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
 $scope.reset = () => { $scope.board = Array(9).fill(''); $scope.turn = 'X'; $scope.winner = ''; };
 $scope.play = i => !$scope.board[i] && !$scope.winner && ($scope.board[i] = $scope.turn, $scope.turn = $scope.turn === 'X' ?
'O' : 'X', win.some(p => {
 if ($scope.board[p[0]] && p.every(j => $scope.board[j] === $scope.board[p[0]])) $scope.winner = $scope.board[p[0]];
 }));
 $scope.reset();
 });
 </script>
</body>
</html>
7. Simple HealthCare using Asp.Net
Open Microsoft Visual Studio Go to File → New → Project
Select ASP.NET Web Application (.NET Framework)
Name it HealthCarePortal
Choose Web Forms template → Click Create
In Solution Explorer, right click your project → Add → Web Form
Name it HealthCare.aspx
HealthCare.aspx
<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="HealthCare.aspx.cs" Inherits="HealthCarePortal.HealthCare"
%>
<!DOCTYPE html>
<html>
<head runat="server">
 <title>Health Care Portal</title>
 <!-- Bootstrap CSS -->
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body class="bg-light">
 <form id="form1" runat="server" class="container mt-5">
 <div class="card shadow p-4">
 <h2 class="text-center text-primary mb-4"> Health Care Portal</h2>
 <div class="mb-3">
 <label for="txtName" class="form-label">Patient Name</label>
 <asp:TextBox ID="txtName" runat="server" CssClass="form-control" />
 </div>
 <div class="mb-3">
 <label for="txtAge" class="form-label">Age</label>
 <asp:TextBox ID="txtAge" runat="server" CssClass="form-control" />
 </div>
 <div class="mb-3">
 <label for="txtDisease" class="form-label">Disease</label>
 <asp:TextBox ID="txtDisease" runat="server" CssClass="form-control" />
 </div>
 <div class="d-grid mb-4">
 <asp:Button ID="btnAdd" runat="server" Text="Add Patient" CssClass="btn btn-success btn-lg" OnClick="btnAdd_Click"
/>
 </div>
 <h4 class="text-secondary">Patient List</h4>
 <asp:GridView ID="GridView1" runat="server" CssClass="table table-bordered table-striped mt-3"
 AutoGenerateColumns="true"></asp:GridView>
 </div>
 </form>
 <!-- Bootstrap JS (optional) -->
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
HealthCare.aspx.cs
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Xml.Linq;
namespace HealthCarePortal
{
 public partial class HealthCare : System.Web.UI.Page
 {
 static DataTable dt = new DataTable();
 protected void Page_Load(object sender, EventArgs e)
 {
 if (!IsPostBack)
 {
 dt.Columns.Add("Name");
 dt.Columns.Add("Age");
 dt.Columns.Add("Disease");
 GridView1.DataSource = dt;
 GridView1.DataBind();
 }
 }
 protected void btnAdd_Click(object sender, EventArgs e)
 {
 DataRow dr = dt.NewRow();
 dr["Name"] = txtName.Text;
 dr["Age"] = txtAge.Text;
 dr["Disease"] = txtDisease.Text;
 dt.Rows.Add(dr);
 GridView1.DataSource = dt;
 GridView1.DataBind();
 // clear fields
 txtName.Text = "";
 txtAge.Text = "";
 txtDisease.Text = "";
 }
 }
}
8. Online Resume using Asp.Net
Open Microsoft Visual Studio Go to File → New → Project
Select ASP.NET Web Application (.NET Framework)
Name it as Resume, Choose Web Forms template → Click Create
In Solution Explorer, right click your project → Add → Web Form
Name it Resume.aspx
Resume.aspx
<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Resume.aspx.cs" Inherits="Resume.Resume" %>
<!DOCTYPE html>
<html>
<head runat="server">
 <meta charset="utf-8" />
 <title>Online Resume - John Doe</title>
 <style>
 body { font-family: Arial; margin: 30px; background-color: #f2f2f2; }
 .resume { max-width: 800px; margin: auto; padding: 25px; background: #fff; border-radius: 8px; box-shadow: 0 0 12px #aaa; }
 h1 { text-align: center; color: #2c3e50; }
 h2 { color: #1abc9c; border-bottom: 2px solid #1abc9c; padding-bottom: 5px; margin-top: 25px; }
 p { margin: 6px 0; }
 ul { margin: 6px 0 6px 20px; }
 </style>
</head>
<body>
 <form id="form1" runat="server">
 <div class="resume">
 <h1><asp:Label ID="lblName" runat="server" Text="John Doe"></asp:Label></h1>
 <p><strong>Email:</strong> <asp:Label ID="lblEmail" runat="server" Text="john.doe@example.com"></asp:Label></p>
 <p><strong>Phone:</strong> <asp:Label ID="lblPhone" runat="server" Text="+91-9876543210"></asp:Label></p>
 <p><strong>Location:</strong> <asp:Label ID="lblLocation" runat="server" Text="Bangalore, India"></asp:Label></p>
 <h2>Education</h2>
 <p><asp:Label ID="lblEducation" runat="server"
 Text="B.Tech in Computer Science, XYZ University, 2024 (CGPA: 8.7/10)"></asp:Label></p>
 <h2>Skills</h2>
 <ul>
 <li><asp:Label ID="lblSkill1" runat="server" Text="C#, ASP.NET Web Forms & MVC"></asp:Label></li>
 <li><asp:Label ID="lblSkill2" runat="server" Text="SQL Server, Database Design"></asp:Label></li>
 <li><asp:Label ID="lblSkill3" runat="server" Text="HTML, CSS, JavaScript, Bootstrap"></asp:Label></li>
 <li><asp:Label ID="lblSkill4" runat="server" Text="Problem Solving & Debugging"></asp:Label></li>
 </ul>
 <h2>Projects</h2>
 <ul>
 <li><asp:Label ID="lblProject1" runat="server" Text="HealthCare Portal – Built with ASP.NET for patient record
management."></asp:Label></li>
 <li><asp:Label ID="lblProject2" runat="server" Text="Online Shopping Cart – Implemented product listing, cart, and order
processing."></asp:Label></li>
 <li><asp:Label ID="lblProject3" runat="server" Text="Resume Builder App – Designed a web app to generate resumes
online."></asp:Label></li>
 </ul>
 <h2>Experience</h2>
 <p><asp:Label ID="lblExperience" runat="server"
 Text="Intern at ABC Tech Solutions (Jan 2024 – June 2024) – Assisted in developing ASP.NET web applications and testing
modules."></asp:Label></p>
 <h2>Achievements</h2>
 <ul>
 <li>Ranked 2nd in University Coding Challenge 2023</li>
 <li>Organized National Level TechFest at XYZ University</li>
 <li>Winner of Smart India Hackathon 2022 (Team Project)</li>
 </ul>
 <h2>Hobbies</h2>
 <ul>
 <li>Open-source coding contributions</li>
 <li>Reading technology blogs</li>
 <li>Playing chess</li>
 </ul>
 </div>
 </form> <!-- <- IMPORTANT: closing form tag -->
</body>
</html>

10. Comet Effect
Create a index.html file and simply paste the code and copy the file path and then paste it in the browser
Index.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8" />
 <title>Comet Effect</title>
 <style>
 body, html { margin: 0; padding: 0; overflow: hidden; background: black; }
 canvas { display: block; }
 </style>
</head>
<body>
 <script src="https://cdn.jsdelivr.net/gh/soulwire/sketch.js@master/js/sketch.min.js"></script>
 <script>
 var particles = [];
 var lastX = 0, lastY = 0;
 function Particle(x, y, dx, dy) {
 this.x = x;
 this.y = y;
 this.vx = dx * 0.2 + (Math.random() - 0.5) * 1.5;
 this.vy = dy * 0.2 + (Math.random() - 0.5) * 1.5;
 this.alpha = 1;
 this.size = Math.random() * 5 + 3;
 }
 Particle.prototype.update = function() {
 this.x += this.vx;
 this.y += this.vy;
 this.alpha -= 0.02;
 this.size *= 0.96;
 };
 Particle.prototype.draw = function(ctx) {
 ctx.fillStyle = 'rgba(255, 255, 200, ' + this.alpha + ')';
 ctx.beginPath();
 ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
 ctx.fill();
 };
 Sketch.create({
 setup: function() {
 this.background = '#000';
 lastX = this.width / 2;
 lastY = this.height / 2;
 },
 draw: function() {
 // Motion blur background
 this.context.fillStyle = 'rgba(0, 0, 0, 0.2)';
 this.context.fillRect(0, 0, this.width, this.height);
 // Only spawn particles when mouse is inside canvas
 if (this.mouse.x !== lastX || this.mouse.y !== lastY) {
 var dx = this.mouse.x - lastX;
 var dy = this.mouse.y - lastY;
 for (var i = 0; i < 4; i++) {
 particles.push(new Particle(this.mouse.x, this.mouse.y, dx, dy));
 }
 lastX = this.mouse.x;
 lastY = this.mouse.y;
 }
 // Update + draw particles
 for (var i = particles.length - 1; i >= 0; i--) {
 var p = particles[i];
 p.update();
 if (p.alpha <= 0 || p.size <= 0.5) {
 particles.splice(i, 1);
 } else {
 p.draw(this.context);
 }
 }
 // Bright glowing comet head
 if (this.mouse.x && this.mouse.y) {
 var gradient = this.context.createRadialGradient(
 this.mouse.x, this.mouse.y, 0,
 this.mouse.x, this.mouse.y, 60
 );
 gradient.addColorStop(0, 'rgba(255,255,255,1)');
 gradient.addColorStop(0.3, 'rgba(255,200,150,0.8)');
 gradient.addColorStop(1, 'rgba(255,200,150,0)');
 this.context.fillStyle = gradient;
 this.context.beginPath();
 this.context.arc(this.mouse.x, this.mouse.y, 60, 0, Math.PI * 2);
 this.context.fill();
 }
 }
 });
 </script>
</body>
</html>