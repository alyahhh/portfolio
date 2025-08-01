import 'package:flutter/material.dart';

void main() {
  runApp(MaterialApp(
    home: Scaffold(
      appBar: AppBar(
        title: Text('I Am Poor',
        style:TextStyle(
          color: Colors.black,
          fontFamily: 'Arial',
          fontSize: 36,
          fontWeight: FontWeight.bold,
        ),),
        backgroundColor: Colors.amber,
      ),
      body: Center(
        child: Image(
          image: AssetImage('images/rock.png'
          ),
          width: 500,
          height: 500,
          fit: BoxFit.contain,
        ),
      ),
      backgroundColor: Colors.amberAccent,
    ),
  ));
}
