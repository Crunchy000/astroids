package games.planetwars.runners

import competition_entry.GreedyHeuristicAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.evo.SimpleEvoAgent
import games.planetwars.agents.random.BetterRandomAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.agents.strategic.TeamTitansAgentV3
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player

// Plays two agents against each other on league-style maps and prints one RESULT line.
// An agent is a sample agent of their repo (pure, better, careful, greedy, evo, evo-big, titans) or a server
// already listening on a local port (remote<PORT>).
// Usage: A,B,SETS,GAMES_PER_SIDE,TIMEOUT_MS   e.g. remote8410,titans,8,2,200
// Every set is a fresh GameParamGenerator.randomParams(); each set plays GAMES_PER_SIDE games from each side.
// Output: RESULT a=<A> b=<B> a_wins=<points of A> b_wins=<points of B> games=<n> a_timeouts=.. b_timeouts=.. a_ms=.. b_ms=..
fun agentFor(key: String): PlanetWarsAgent = when (key) {
    "pure" -> PureRandomAgent()
    "better" -> BetterRandomAgent()
    "careful" -> CarefulRandomAgent()
    "greedy" -> GreedyHeuristicAgent()
    "evo" -> SimpleEvoAgent()
    "evo-big" -> SimpleEvoAgent(nEvals = 60, timeLimitMillis = 40)
    "titans" -> TeamTitansAgentV3()
    else -> if (key.startsWith("remote")) RemoteAgent("<unused>", port = key.removePrefix("remote").toInt()) else error("unknown agent $key")
}

fun main(args: Array<String>) {
    val p = args[0].split(',')
    val keyA = p[0]
    val keyB = p[1]
    val sets = p[2].toInt()
    val perSide = p[3].toInt()
    val timeout = p[4].toLong()
    val a = agentFor(keyA)
    val b = agentFor(keyB)
    if (a is RemoteAgent) waitForAgentType(a)
    if (b is RemoteAgent) waitForAgentType(b)
    var aPoints = 0.0
    var bPoints = 0.0
    var games = 0
    var aTimeouts = 0
    var bTimeouts = 0
    var aMs = 0.0
    var bMs = 0.0
    for (s in 1..sets) {
        val league = RoundRobinLeague(
            agents = listOf(a, b), gameParams = GameParamGenerator.randomParams(), gamesPerPair = perSide,
            runRemoteAgents = true, timeout = timeout,
        )
        // a plays Player1, then Player2; the names are only used for logging
        val first = league.runPair(a, b, "A", "B")
        val second = league.runPair(b, a, "B", "A")
        aPoints += first.first[Player.Player1]!! + second.first[Player.Player2]!!
        bPoints += first.first[Player.Player2]!! + second.first[Player.Player1]!!
        aTimeouts += first.third[Player.Player1]!! + second.third[Player.Player2]!!
        bTimeouts += first.third[Player.Player2]!! + second.third[Player.Player1]!!
        aMs += (first.second[Player.Player1]!! + second.second[Player.Player2]!!) / 2
        bMs += (first.second[Player.Player2]!! + second.second[Player.Player1]!!) / 2
        games += 2 * perSide
        println("PROGRESS set $s/$sets: $aPoints-$bPoints of $games")
    }
    println("RESULT a=$keyA b=$keyB a_wins=${aPoints} b_wins=${bPoints} games=$games a_timeouts=$aTimeouts b_timeouts=$bTimeouts " +
        "a_ms=${"%.2f".format(aMs / sets)} b_ms=${"%.2f".format(bMs / sets)}")
    System.exit(0)
}
