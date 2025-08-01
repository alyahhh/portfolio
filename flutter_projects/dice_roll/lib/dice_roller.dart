import "dart:math";

import 'package:flutter/material.dart';

final randomizer = Random();

class DiceRoller extends StatefulWidget {
  const DiceRoller({super.key});

  @override
  State<DiceRoller> createState() {
    return _DiceRollerState();
  }
}

class _DiceRollerState extends State<DiceRoller> {
  var currentDiceRoll = 2;

  void rollDice() {
    setState(() {
      currentDiceRoll = randomizer.nextInt(6) + 1;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image.asset('assets/images/dice-$currentDiceRoll.png', width: 200),
        const SizedBox(height: 15),
        OutlinedButton(
          onPressed: rollDice,
          style: OutlinedButton.styleFrom(
            side: const BorderSide(width: 2, color: Colors.black),
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            foregroundColor: Colors.black,
            textStyle: const TextStyle(fontSize: 20),
          ),
          child: const Text('Roll Dice'),
        ),
      ],
    );
  }
}
