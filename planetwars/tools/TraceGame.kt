package games.planetwars.runners

import games.planetwars.agents.Action
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.strategic.TeamTitansAgentV3
import games.planetwars.core.GameParamGenerator
import games.planetwars.core.Player
import java.io.File
import kotlin.math.sqrt

// Plays one game of a remote agent against TeamTitansAgentV3 and writes a JSON-lines trace:
// a header with the map, every launch, and a snapshot every 10 ticks.
// Usage: PORT,SIDE,OUTFILE   (SIDE 1: remote plays Player1)
fun main(args: Array<String>) {
    val p = args[0].split(',')
    val out = File(p[2]).bufferedWriter()
    val remote = RemoteAgent("<unused>", port = p[0].toInt())
    waitForAgentType(remote)
    val titans = TeamTitansAgentV3()
    val params = GameParamGenerator.randomParams()
    val remoteIsP1 = p[1] == "1"
    val runner = if (remoteIsP1) GameRunner(remote, titans, params) else GameRunner(titans, remote, params)
    val fm = runner.forwardModel
    val ps0 = fm.state.planets
    out.write("{\"type\":\"header\",\"remote_side\":${if (remoteIsP1) 1 else 2},\"speed\":${params.transporterSpeed},\"planets\":[" +
        ps0.joinToString(",") { "{\"id\":${it.id},\"x\":${it.position.x},\"y\":${it.position.y},\"growth\":${it.growthRate},\"radius\":${it.radius},\"owner\":${it.owner.ordinal},\"ships\":${it.nShips}}" } + "]}\n")
    fun side(pl: Player) = if (pl == Player.Player1) 1 else 2
    var t = 0
    while (!fm.isTerminal()) {
        val s = fm.state
        val acts = mapOf(Player.Player1 to runner.agent1.getAction(s), Player.Player2 to runner.agent2.getAction(s))
        for ((pl, a) in acts) {
            if (a.playerId != Player.Neutral && a.sourcePlanetId >= 0) {
                val src = s.planets[a.sourcePlanetId]; val dst = s.planets[a.destinationPlanetId]
                val d = sqrt((src.position.x - dst.position.x) * (src.position.x - dst.position.x) + (src.position.y - dst.position.y) * (src.position.y - dst.position.y))
                out.write("{\"type\":\"launch\",\"t\":$t,\"p\":${side(pl)},\"src\":${a.sourcePlanetId},\"dst\":${a.destinationPlanetId},\"n\":${a.numShips},\"src_ships\":${src.nShips},\"dst_owner\":${dst.owner.ordinal},\"dst_ships\":${dst.nShips},\"dst_growth\":${dst.growthRate},\"dist\":$d}\n")
            }
        }
        if (t % 10 == 0) {
            fun tot(pl: Player) = s.planets.filter { it.owner == pl }
            fun fleet(pl: Player) = s.planets.mapNotNull { it.transporter }.filter { it.owner == pl }.sumOf { it.nShips }
            out.write("{\"type\":\"snap\",\"t\":$t," + listOf(Player.Player1, Player.Player2).joinToString(",") { pl ->
                "\"p${side(pl)}\":{\"planets\":${tot(pl).size},\"growth\":${"%.3f".format(tot(pl).sumOf { it.growthRate })},\"ships\":${"%.1f".format(tot(pl).sumOf { it.nShips })},\"fleet\":${"%.1f".format(fleet(pl))}}"
            } + ",\"neutral\":${s.planets.count { it.owner == Player.Neutral }}}\n")
        }
        fm.step(acts)
        t++
    }
    out.write("{\"type\":\"final\",\"t\":$t,\"p1\":${fm.getShips(Player.Player1)},\"p2\":${fm.getShips(Player.Player2)},\"leader\":${side(fm.getLeader().let { if (it == Player.Neutral) Player.Player1 else it })}}\n")
    out.close()
    println("FINAL tick $t: " + fm.statusString() + " remote_is_p${if (remoteIsP1) 1 else 2}")
}
