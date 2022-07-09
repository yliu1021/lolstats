window.onload = function () {
    let summonerInput = document.getElementById("summonerInput")
    let matchSummary = document.getElementById("matchSummary")

    function clearSummary() {
        while (matchSummary.firstChild) {
            matchSummary.removeChild(matchSummary.firstChild)
        }
    }

    function setLoading() {
        clearSummary()
        const loading = document.createElement("p")
        loading.innerText = "Loading..."
        matchSummary.appendChild(loading)
    }

    function showError(msg) {
        clearSummary()
        const error = document.createElement("p")
        error.innerText = msg
        matchSummary.appendChild(error)
    }

    function showPrediction(chancesOfWinning, game) {
        clearSummary()
        chancesOfWinning *= 100
        const winTextField = document.createElement("p")
        winTextField.innerHTML = `Winning chances: <b>${chancesOfWinning.toFixed(2)}%</b>`
        winTextField.classList.add("text-lg")
        matchSummary.appendChild(winTextField)

        function playerDiv(player) {
            const div = document.createElement("div")
            const name = document.createElement("p")
            const champion = document.createElement("p")
            name.innerText = player["summonerName"]
            name.classList.add("font-bold")
            champion.innerText = player["champion"]
            champion.classList.add("italic")
            div.appendChild(name)
            div.appendChild(champion)
            return div
        }

        function showTeam(team) {
            const div = document.createElement("div")
            team.forEach((player, ind) => {
                div.appendChild(playerDiv(player))
            })
            return div
        }

        const gameDiv = document.createElement("div")
        gameDiv.classList.add("flex", "flex-row", "justify-between")
        const team1 = showTeam(game["team1"])
        const team2 = showTeam(game["team2"])
        team1.classList.add("text-left")
        team2.classList.add("text-right")
        gameDiv.appendChild(team1)
        gameDiv.appendChild(team2)
        matchSummary.append(gameDiv)
    }

    function unknownError() {
        showError("Unknown error occured")
    }

    function search(summonerName) {
        setLoading()
        fetch('live?' + new URLSearchParams({ summoner: summonerName }))
            .then(res => res.json())
            .then(res => {
                if (res["error"]) {
                    showError(res["error"])
                } else if (res["chancesOfWinning"] && res["game"]) {
                    showPrediction(res["chancesOfWinning"], res["game"])
                } else {
                    unknownError()
                }
            })
            .catch(unknownError)
    }

    summonerInput.onkeydown = function (e) {
        if (e.key === 'Enter') {
            search(e.target.value)
        }
    }
}
