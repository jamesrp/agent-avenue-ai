# Game rules
These are the rules for the basic two-player game.
## Setup
Shuffle the agent cards and deal each player 4 cards in hand. Put the
remaining cards in a face-down draw pile. It is called the agent deck.

## Winning
Player scores each start out at 0. A player wins if they get 7 or more
points higher than their opponent. There are also two cards (Codebreaker
and Daredevil) that can cause an instant win or instant loss.

## Game turns
The players alternate turns. On your turn, perform the following steps
in this order:
1. Play
2. Recruit
3. End

### Play
Play 2 cards from your hand in the center of the table, 1 face up and 1
face down. They must have different card names. (In the rare
case that all the cards in your hand have the same card name, it is
allowed to play 2 identical cards.)

Draw cards from the agent deck until you have 4 cards in
hand. Do this only after you have played both cards.

### Recruit
The opponent must recruit 1 of the 2 agent cards you have played. To do
so, they choose 1 card and put it face up in front of them. You must now
recruit the other agent card by putting it face up in front of you.

Then both players adjust their score based on the agent they recruited.
Check how many agent cards with that same name you have in play,
including the one you have just recruited.
If you have only 1, look at its 1st icon on the top.
If you have 2, look at its 2nd icon in the center instead.
If you have 3 or more, look at its 3rd icon on the bottom instead.

### End
After both meeples have moved, check whether any player fulfills a
condition to win or lose the game. If that is not the case, the game
continues, your turn ends, and your opponent’s turn begins.
* A player with score >= opponent score + 7 wins the game
* A player with 3 codebreakers wins the game
* A player with 3 daredevils loses the game

If neither player won or lost the game and the agent deck
is empty and your opponent has less than 2 cards in hand, the game ends
(because your opponent can’t play next turn). In that case, the player
with a higher score wins the game.

In the event of a tie, the player whose turn it is (the active player)
wins the game. This occurs in the following cases:
* Both players fulfill a condition to win the game.
* Both players fulfill a condition to lose the game.
* Any player fulfills a condition to win the game and a condition to lose the game.
* Players run out of cards (see above) and have the same score.

## Card Contents
The deck has 38 agent cards. There are 6 each of:
* Double Agent (-1, 6, -1)
* Enforcer (1, 2, 3)
* Codebreaker (0, 0, WIN)
* Daredevil (2, 3, LOSE)
* Saboteur (-1, -1, -2)
* Sentinel (0, 2, 6)

The tuple indicates the effect on first, second, or third+ copy.
For instance, Sentinel (0, 2, 6) means:
* First copy gains 0 points
* Second copy gains 2 points
* Third copy (or fourth etc) gains 6 points

There are also 1 each of:
* Sidekick (4)
* Mole (-3)

Since there is only one each of these, they don't have a 2nd-copy or 3rd-copy effect.
