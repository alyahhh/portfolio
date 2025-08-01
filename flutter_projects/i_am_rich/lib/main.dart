import 'package:flutter/material.dart';

// The main function is the starting point of all flutter apps.
void main() {
  runApp(MaterialApp(
    home: Scaffold(
      backgroundColor: Colors.blueGrey,
      appBar: AppBar(
        title: Text('I Am Rich',
        style: TextStyle(
          fontFamily: 'Arial',
          fontSize: 36,
          fontWeight: FontWeight.bold,
          color: Colors.white,
        ),
        ),
        backgroundColor: Colors.blueGrey[900],
      ),
      body: Center(
        child: Image(
          image: AssetImage('images/diamond.png'),
        ),
      ),
    ),
  ));
}

