from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar, Callable
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9

    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

    def attack(self, target: Unit):
        targetDamage = -self.damage_amount(target)
        selfDamage = -target.damage_amount(self)
        self.mod_health(selfDamage)
        target.mod_health(targetDamage)

        return (True, f"{self} attacks {target} for {self.damage_amount(target)} damage")

    def repair(self, target: Unit):
        healAmount = self.repair_amount(target)
        target.mod_health(healAmount)

        return (True, f"{self} repairs {target} for {self.repair_amount(target)} health")

    def self_destruct(self, targets: list[Unit]):
        for unit in targets:
            unit.mod_health(-2)
        self.mod_health(-9)
        return (True, f"{self} self-destructs")


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    def iter_adjacent_with_diagonals(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col - 1)
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row - 1, self.col + 1)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)
        yield Coord(self.row + 1, self.col + 1)

    def has_adjacent(self, target: Coord) -> bool:
        """Checks if a Coord is adjacent to this one."""
        for adj in self.iter_adjacent():
            if adj == target:
                return True
        return False

    def up(self) -> Coord:
        """Coord above this one."""
        return Coord(self.row - 1, self.col)

    def down(self) -> Coord:
        """Coord below this one."""
        return Coord(self.row + 1, self.col)

    def left(self) -> Coord:
        """Coord to the left of this one."""
        return Coord(self.row, self.col - 1)

    def right(self) -> Coord:
        """Coord to the right of this one."""
        return Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def distance(cls, src: Coord, dst: Coord) -> int:
        """Calculate the distance between two Coords."""
        return abs(src.row - dst.row) + abs(src.col - dst.col)

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    start_time: float = 0.0
    total_seconds: float = 0.0
    cumulative_eval_per_depth: dict[int, int] = field(default_factory=dict)

##############################################################################################################

class Heuristic(Enum):
    e0 = "e0"
    e1 = "e1"
    e2 = "e2"


##############################################################################################################

class AI:
    @staticmethod
    def e0(game: Game) -> int:
        pos : int = 0
        neg : int = 0
        for coord in CoordPair.from_dim(game.options.dim).iter_rectangle():
            unit = game.get(coord)
            if unit is not None:
                if unit.player == Player.Attacker:
                    if unit.type == UnitType.AI: pos += 9999
                    else: pos += 3
                else:
                    if unit.type == UnitType.AI: neg += 9999
                    else: neg += 3
        return pos - neg

    @staticmethod
    def proximity_score(game: Game, player_coord: Coord, enemy_units: Iterable[tuple[Coord, Unit]]):
        """Encourage getting closer to units you have favorable match-up over"""
        score = 0
        player_unit = game.get(player_coord)
        # Firewall board control is priority
        if player_unit.type == UnitType.Firewall:
            distance = CoordPair.distance(player_coord, Coord(2, 2))
            score += 100 / distance + 1
        for enemy_coord, enemy_unit in enemy_units:
            distance = CoordPair.distance(player_coord, enemy_coord)
            damage_amount = player_unit.damage_amount(enemy_unit)
            if (player_unit.type == UnitType.Virus and enemy_unit.type == UnitType.AI):
                score += 4 * damage_amount / distance + 1
            # extra points if a Tech unit is close to the enemy AI
            if player_unit.type == UnitType.Virus and enemy_unit.type == UnitType.Tech:
                score += 2 * damage_amount / distance + 1
            else:
                score += damage_amount / distance

        return score

    @staticmethod
    def health_score(enemy_units: Iterable[tuple[Coord, Unit]]):
        """Calculates a score based on the enemy units' health."""
        score = 0
        for _, unit in enemy_units:
            score -= unit.health
        return score

    @staticmethod
    def e1(game: Game) -> int:
        """Heuristic that calculates the score of the game
        based on the proximity of the player's units to the enemy's units and the health of enemy units."""
        attacker_units = game.player_units(Player.Attacker)
        defender_units = game.player_units(Player.Defender)

        attacker_score = AI.health_score(defender_units)
        defender_score = AI.health_score(attacker_units)

        return attacker_score - defender_score

    @staticmethod
    def e2_unitscore(game: Game, unit : Unit, coord: Coord) -> int:
        if unit.type == UnitType.AI:
            return 10*(unit.health + 1)
        elif unit.type == UnitType.Firewall:
            center: int = game.options.dim/2
            rowDist = 1 - abs(coord.row - center)/game.options.dim
            colDist = 1 - abs(coord.col - center)/game.options.dim
            return (rowDist + colDist) / 4 + (unit.health+1)/2
        elif unit.type == UnitType.Tech:
            points = 0
            for adj in coord.iter_adjacent():
                adjacent = game.get(adj)
                if adjacent is not None and adjacent.type == UnitType.Firewall and unit.player == adjacent.player:
                    points += (adjacent.health+1)
            
            return max(points,10)
        elif unit.type == UnitType.Virus:
            points = 0
            for adj in coord.iter_adjacent():
                adjacent = game.get(adj)
                if adjacent is not None and unit.player != adjacent.player:
                    if adjacent.type == UnitType.AI:
                        points += 1/(adjacent.health+1) * 10
                    elif adjacent.type == UnitType.Tech:
                        points += 1/(adjacent.health+1) * 5
            return points
        elif unit.type == UnitType.Program:
            points = 0
            for adj in coord.iter_adjacent():
                adjacent = game.get(adj)
                if adjacent is not None and unit.player != adjacent.player:
                    if adjacent.type == UnitType.AI:
                        points += 1/(adjacent.health+1) * 10
                    elif adjacent.type == UnitType.Virus:
                        points += 1/(adjacent.health+1) * 5
                    elif adjacent.type == UnitType.Tech:
                        points += 1/(adjacent.health+1) * 5
            return points
        return 0

    @staticmethod
    def e2(game: Game) -> int:
        pos : int = 0
        neg : int = 0
        for coord in CoordPair.from_dim(game.options.dim).iter_rectangle():
            unit = game.get(coord)
            if unit is not None:
                if unit.player == Player.Attacker:
                    pos += AI.e2_unitscore(game, unit, coord)
                else:
                    neg += AI.e2_unitscore(game, unit, coord)
        return pos - neg

    @staticmethod
    def call_heuristic(heuristic: Heuristic, game: Game) -> int:
        """Calls the heuristic function based on the enum value."""
        match heuristic:
            case Heuristic.e0:
                return AI.e0(game)
            case Heuristic.e1:
                return AI.e1(game)
            case Heuristic.e2:
                return AI.e2(game)

    @staticmethod
    def mini_max(game: Game, depth: int,  heuristic: Heuristic, is_maximizing: bool = True) \
            -> Tuple[int, CoordPair | None, float]:
        # recursive end condition: either we reach max depth or find a game winning move.
        if game.is_finished() or depth == 0:
            return AI.call_heuristic(heuristic, game), None, 0

        # Define Variables
        moves = game.move_candidates()  # all possible moves this turn
        best_move = None  # best possible move
        best_score = 0  # score associated with best move

        # set the best score depending on if we're maximizing or minimizing
        if is_maximizing:   best_score = MIN_HEURISTIC_SCORE
        else:               best_score = MAX_HEURISTIC_SCORE

        # loop through each possible move
        for move in moves:
            # check for the validity of the move
            game_clone = game.clone()
            is_valid, _ = game_clone.perform_move(move, False)
            if is_valid:
                # progress the game
                game_clone.next_turn()
                # evaluate the next series of moves
                (score, _, _) = AI.mini_max(game_clone, depth - 1, heuristic, not is_maximizing)

                # update max score based on if we're maximizing or minimizing
                if (is_maximizing and score > best_score) or (not is_maximizing and score < best_score):
                    best_score = score
                    best_move = move

                #if we're about to exceed maximum time, cut our losses and return our best guess.
                if (datetime.now() - game.stats.start_time).total_seconds() + 0.1 >= game.options.max_time:
                    break

                # update stats
                game.stats.evaluations_per_depth[game.options.max_depth - depth] = game.stats.evaluations_per_depth.get(game.options.max_depth - depth, 0) + 1

        # we've looped through all the moves, return the best one
        return best_score, best_move, 0

    @staticmethod
    def alpha_beta(game: Game, depth: int,  heuristic: Heuristic,  is_maximizing: bool = True, alpha :int = MIN_HEURISTIC_SCORE, beta: int = MAX_HEURISTIC_SCORE) \
            -> Tuple[int, CoordPair | None, float]:
        # recursive end condition: either we reach max depth or find a game winning move.
        if game.is_finished() or depth == 0:
            return AI.call_heuristic(heuristic, game), None, 0

        # Define Variables
        moves = game.move_candidates()  # all possible moves this turn
        best_move = None  # best possible move
        best_score = 0  # score associated with best move

        # set the best score depending on if we're maximizing or minimizing
        if is_maximizing:   best_score = MIN_HEURISTIC_SCORE
        else:               best_score = MAX_HEURISTIC_SCORE

        # loop through each possible move
        for move in moves:
            # check for the validity of the move
            game_clone = game.clone()
            is_valid, _ = game_clone.perform_move(move, False)
            if is_valid:
                # progress the game
                game_clone.next_turn()
                # evaluate the next series of moves
                (score, _, _) = AI.alpha_beta(game_clone, depth - 1, heuristic, not is_maximizing, alpha, beta)

                # update max score based on if we're maximizing or minimizing and update constraints
                if is_maximizing:
                    alpha = max(alpha, best_score)
                    if score > best_score:
                        best_score = score
                        best_move = move
                else:
                    beta = min(beta, best_score)
                    if score < best_score:
                        best_score = score
                        best_move = move

                #enforce constraints
                if beta <= alpha:
                    break  # prune

                #if we're about to exceed maximum time, cut our losses and return our best guess.
                if (datetime.now() - game.stats.start_time).total_seconds() + 0.1 >= game.options.max_time:
                    break

                # update stats
                game.stats.evaluations_per_depth[game.options.max_depth - depth] = game.stats.evaluations_per_depth.get(game.options.max_depth - depth, 0) + 1

        # we've looped through all the moves, return the best one
        return best_score, best_move, 0


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic: Heuristic = Heuristic.e0


##############################################################################################################
@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))


    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def is_in_combat(self, coord: Coord) -> bool:
        """Check if unit at Coord is in combat."""
        unit = self.get(coord)
        if unit is not None:
            for adj in coord.iter_adjacent():
                target = self.get(adj)
                if target is not None and target.player != unit.player:
                    return True
        return False

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        current_unit = self.get(coords.src)
        target_unit = self.get(coords.dst)

        if current_unit is None or current_unit.player != self.next_player:
            return False

        # self-destruct
        if coords.src == coords.dst and current_unit.player == target_unit.player:
            return True

        if not coords.src.has_adjacent(coords.dst):
            return False

        # attacking / repairing
        if target_unit is not None:
            if target_unit.player == current_unit.player and target_unit.health == 9:
                return False
            return True

        # moving
        match self.next_player:
            case Player.Attacker:
                if current_unit.type == UnitType.AI or current_unit.type == UnitType.Firewall or current_unit.type == UnitType.Program:
                    if self.is_in_combat(coords.src):
                        # print("The unit is in combat and can't move!")
                        return False
                    if coords.dst == coords.src.up() or coords.dst == coords.src.left():
                        return True
                if current_unit.type == UnitType.Tech or current_unit.type == UnitType.Virus:
                    for adj in coords.src.iter_adjacent():
                        if adj == coords.dst:
                            return True
            case Player.Defender:
                if current_unit.type == UnitType.AI or current_unit.type == UnitType.Firewall or current_unit.type == UnitType.Program:
                    if self.is_in_combat(coords.src):
                        # print("The unit is in combat and can't move!")
                        return False
                    if coords.dst == coords.src.down() or coords.dst == coords.src.right():
                        return True
                if current_unit.type == UnitType.Tech or current_unit.type == UnitType.Virus:
                    for adj in coords.src.iter_adjacent():
                        if adj == coords.dst:
                            return True

        return False

    def get_adjacent_units_with_diagonals(self, coord: Coord) -> list[Unit]:
        neighbors: list[Unit] = []
        for adj in coord.iter_adjacent_with_diagonals():
            neighbor = self.get(adj)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

    def perform_move(self, coords: CoordPair, is_real_move: bool = False, score: int = 0) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""
        if self.is_valid_move(coords):
            current_unit = self.get(coords.src)
            target_unit = self.get(coords.dst)

            if current_unit is None:
                return (False, "invalid move")

            if coords.src == coords.dst:
                output = current_unit.self_destruct(self.get_adjacent_units_with_diagonals(coords.src))
                for cell in coords.src.iter_adjacent_with_diagonals():
                    self.remove_dead(cell)
                self.remove_dead(coords.src)
                if (is_real_move == True):
                    self.output_file_midgame(output[1], score)
                return output

            if target_unit is not None:
                if target_unit.player is not current_unit.player:
                    output = current_unit.attack(target_unit)
                    self.remove_dead(coords.src)
                    self.remove_dead(coords.dst)
                    if (is_real_move == True):
                        self.output_file_midgame(output[1], score)
                    return output
                elif target_unit.player is current_unit.player:
                    output = current_unit.repair(target_unit)
                    if (is_real_move == True):
                        self.output_file_midgame(output[1], score)
                    return output

            # just moving
            self.set(coords.dst, self.get(coords.src))
            if (is_real_move == True):
                output = f"Moved {self.get(coords.src)} from: {coords.src} to: {coords.dst}\n"
                self.output_file_midgame(output, score)
            self.set(coords.src, None)
            return (True, "")
        return (False, "invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv, True)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv, True)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv, score = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv, True, score)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
            else:
                #If an AI gives an invalid move, end the game.
                print("Computer has outputted an invalid move, this should never happen!!!")
                exit(1)     
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return 0, move_candidates[0], 1
        else:
            return 0, None, 0

    def ai_move(self) -> Tuple[int, CoordPair | None, float]:
        move_candidates = list(self.move_candidates())
        if len(move_candidates) > 0:
            if self.options.alpha_beta:
                return AI.alpha_beta(self.clone(), self.options.max_depth, self.options.heuristic, self.next_player == Player.Attacker)
            else:
                return AI.mini_max(self.clone(), self.options.max_depth, self.options.heuristic, self.next_player == Player.Attacker)
        else:
            return 0, None, 0

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta."""
        self.stats.start_time = datetime.now()

        if self.options.randomize_moves:
            (score, move, avg_depth) = self.random_move()
        else:
            (score, move, avg_depth) = self.ai_move()
        elapsed_seconds = (datetime.now() - self.stats.start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        #print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move, score

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def output_file_initial(self):
        b = self.options.alpha_beta
        t = self.options.max_time
        m = self.options.max_turns
        file_name = f"gameTrace-{b}-{t}-{m}.txt"
        try:
            with open(file_name, "w") as file:
                l1 = f"The value of the timeout in seconds is: {t} \n"
                l2 = f"The max number of turns is: {m} \n"
                l3 = f"Alpha-Beta enabled?: {b} \n"
                l4 = f"Play mode is: {self.options.game_type}\n"
                l5 = f"Heuristic used {self.options.heuristic}\n"
                l6 = f"Initial configuration\n"
                l7 = f"{Game.to_string(self)}"
                file.writelines([l1, l2, l3, l4, l5, l6, l7])
        except IOError as e:
            print(f"Error: {e}")

    def output_file_midgame(self, output, score):
        b = self.options.alpha_beta
        t = self.options.max_time
        m = self.options.max_turns
        file_name = f"gameTrace-{b}-{t}-{m}.txt"
        depthEval = ""
        depthEvalPercent = ""
        l1 = f"Turn: {self.turns_played}\n"
        l2 = f"Current Player: {self.next_player.name} \n"
        l3 = f"Action Taken:{output} \n"

        if self.options.game_type == GameType.CompVsComp or (self.options.game_type == GameType.AttackerVsComp and self.next_player == Player.Defender) or (self.options.game_type == GameType.CompVsDefender and self.next_player == Player.Attacker):
            l4 = f"Time for this action: {self.stats.total_seconds} \n"
            l5 = f"Heuristic score: {score}\n"
            for k in sorted(self.stats.evaluations_per_depth.keys()):
                self.stats.cumulative_eval_per_depth[k] =+ self.stats.evaluations_per_depth.get(k,0)
                depthEval += f"Depth level {k} has cumulative evals: {self.stats.cumulative_eval_per_depth[k]}, \n"
                depthEvalPercent += f"Depth level {k} has {self.stats.cumulative_eval_per_depth[k] / sum(self.stats.cumulative_eval_per_depth.values())*100}% of states\n"
            l6 = f"The number of states evaluated since the beginning of the game: {sum(self.stats.cumulative_eval_per_depth.values())} \n"
            l7 = "The number of states evaluated by depth: \n" + depthEval
            l8 = "The cumulative evals by depth in Percentage: \n" + depthEvalPercent
            values = list(self.stats.cumulative_eval_per_depth.values())
            total_non_leaf_nodes = sum(values[:-1]) #exclude the last depth level (leaf nodes)
            l9 = f"Average Branching factor: {sum(self.stats.cumulative_eval_per_depth.values()) / total_non_leaf_nodes} \n"
        else:
            l4, l5, l6, l7, l8, l9 = "", "", "", "", "", ""
        l10 = f"{Game.to_string(self)}"
        try:
            with open(file_name, "a") as file:
                file.writelines([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
        except IOError as e:
            print(f"Error: {e}")

    def output_file_endgame(self):
        b = self.options.alpha_beta
        t = self.options.max_time
        m = self.options.max_turns
        file_name = f"gameTrace-{b}-{t}-{m}.txt"
        try:
            with open(file_name, "a") as file:
                file.writelines(f"{self.has_winner()} wins in {self.turns_played} turns")
        except IOError as e:
            print(f"Error: {e}")


##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game (should be commented out in the final version, this is for testing)
    game = Game(options=options)
    #options.game_type = GameType.AttackerVsComp
    #options.alpha_beta = False
    #options.randomize_moves = False
    #options.heuristic = Heuristic.e2


    game.output_file_initial()
    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            game.output_file_endgame()
            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################

if __name__ == '__main__':
    main()
