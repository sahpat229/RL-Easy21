import numpy as np
import random

class State():
    def __init__(self, player_score=None, dealer_score=None, terminated=None):
        self.player_score = player_score
        self.dealer_score = dealer_score
        self.terminated = terminated

class GameInstance():
    def __init__(self):
        self.deck = Deck(1/3, 2/3)
        self.player = Human()
        self.dealer = Human()

    def humanTurn(self, hit, human):
        if hit and human.hittable:
            continuable = human.hit(self.deck)
        elif hit and not human.hittable:
            continuable = False
        elif not hit:
            continuable = human.stick()
        return continuable

    def playerTurn(self, hit):
        return self.humanTurn(hit, self.player)

    def dealerTurn(self):
        while self.dealer.hittable and self.dealer.score < 17:
            self.dealer.hit(self.deck)
        self.dealer.stick()

    def reward(self):
        if self.player.hittable:
            return 0

        if self.player.busted:
            return -1
        elif self.dealer.busted:
            return 1
        elif self.player.score < self.dealer.score:
            return -1
        elif self.player.score > self.dealer.score:
            return 1
        else:
            return 0

    def step(self, hit):
        continuable = self.playerTurn(hit)

        if continuable:
            print("Not terminated")
            return [State(self.player.score, self.dealer.score, False), self.reward()]
        else:
            print("Terminated")
            if not self.player.busted:
                self.dealerTurn()
            return [State(self.player.score, self.dealer.score, True), self.reward()]

class Human():
    def __init__(self):
        self.cards = []
        self.cards.append(Card("black", random.randint(1, 10)))
        self.hittable = True
        self.busted = False
        self.score = self.cards[0].number

    def hit(self, deck):
        if not self.hittable:
            return

        card = deck.sample()
        self.cards.append(card)
        if card.color == "red":
            self.score -= card.number
        else:
            self.score += card.number

        if not 1 <= self.score <= 21:
            self.busted = True
            self.hittable = False
            return False
        return True

    def stick(self):
        self.hittable = False
        return False

class Card():
    def __init__(self, color, number):
        self.color = color
        self.number = number

class Deck():
    def __init__(self, red_prob, black_prob):
        self.red_prob = red_prob
        self.black_prob = black_prob

    def sample(self):
        card_colors = ["red", "black"]
        return Card(color=card_colors[np.random.choice(2, p=[self.red_prob, self.black_prob])],
                    number=random.randint(1, 10))