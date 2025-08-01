import 'package:flutter/material.dart';

import 'package:dice_roll/gradient_container.dart';

void main() {
  runApp(
    const MaterialApp(
      home: Scaffold(
        body: GradientContainer(
          colors: [
            Color.fromARGB(255, 116, 192, 252),
            Color.fromARGB(255, 216, 180, 254),
          ],
        ),
      ),
    ),
  );
}
