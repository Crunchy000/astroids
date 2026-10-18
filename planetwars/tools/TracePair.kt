package games.planetwars.runners

import games.planetwars.agents.Action
import games.planetwars.agents.RemoteAgent
import games.planetwars.core.ForwardModel
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player
import json_rmi.RemoteConstructable
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.zip.GZIPOutputStream

// Plays two remote agents against each other on seeded league-style maps, each map once from each side,
// and writes the full game state and both moves of every tick to a gzipped JSON-lines file, so a lost game
// can be studied and any position fed back to an agent.
// Usage: A_PORT,B_PORT,FIRST_SEED,MAPS,OUTDIR   e.g. 8551,8552,1000,20,trace/pair
// Prints one line per game: GAME seed=.. a_side=.. ticks=.. a_ships=.. b_ships=.. a_points=..
private val compact = Json(from = json_rmi.json) { prettyPrint = false }

private fun enc(s: GameState) = compact.encodeToString(PolymorphicSerializer(RemoteConstructable::class), s)
private fun enc(a: Action) = compact.encodeToString(Action.serializer(), a)

fun main(args: Array<String>) {
    val p = args[0].split(',')
    val a = RemoteAgent("<unused>", port = p[0].toInt())
    val b = RemoteAgent("<unused>", port = p[1].toInt())
    waitForAgentType(a)
    waitForAgentType(b)
    val first = p[2].toLong()
    val maps = p[3].toInt()
    val dir = File(p[4]).apply { mkdirs() }
    for (seed in first until first + maps) {
        val params = GameParamGenerator.randomParams(seed)
        val start = GameStateFactory(params, seed).createGame()
        for (aSide in listOf(1, 2)) {
            val p1 = if (aSide == 1) a else b
            val p2 = if (aSide == 1) b else a
            p1.prepareToPlayAs(Player.Player1, params)
            p2.prepareToPlayAs(Player.Player2, params)
            val fm = ForwardModel(start.deepCopy(), params)
            val out = GZIPOutputStream(File(dir, "s${seed}_a$aSide.jsonl.gz").outputStream()).bufferedWriter()
            out.write("{\"seed\":$seed,\"a_side\":$aSide,\"speed\":${params.transporterSpeed},\"planets\":${params.numPlanets}}\n")
            while (!fm.isTerminal()) {
                val s = fm.state
                val m1 = p1.getAction(s.deepCopy())
                val m2 = p2.getAction(s.deepCopy())
                out.write("{\"t\":${s.gameTick},\"state\":${enc(s)},\"m1\":${enc(m1)},\"m2\":${enc(m2)}}\n")
                fm.step(mapOf(Player.Player1 to m1, Player.Player2 to m2))
            }
            val aPl = if (aSide == 1) Player.Player1 else Player.Player2
            val bPl = if (aSide == 1) Player.Player2 else Player.Player1
            val leader = fm.getLeader()
            val pts = if (leader == aPl) 1.0 else if (leader == bPl) 0.0 else 0.5
            out.write("{\"final\":true,\"t\":${fm.state.gameTick},\"state\":${enc(fm.state)},\"a_points\":$pts}\n")
            out.close()
            println("GAME seed=$seed a_side=$aSide ticks=${fm.state.gameTick} a_ships=${"%.0f".format(fm.getShips(aPl))} " +
                "b_ships=${"%.0f".format(fm.getShips(bPl))} a_points=$pts")
        }
    }
    System.exit(0)
}
