import { useState, useRef, useEffect } from "react"
import {
  Container,
  Paper,
  Typography,
  TextField,
  IconButton,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Fade,
  Slide,
  Zoom,
  Avatar,
  Chip,
} from "@mui/material"
import {
  Send as SendIcon,
  Psychology as BrainIcon,
  TrendingUp as TrendingUpIcon,
  Person as PersonIcon,
} from "@mui/icons-material"
import { ThemeProvider, createTheme } from "@mui/material/styles"
import CssBaseline from "@mui/material/CssBaseline"
import "./App.css"

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#6366f1",
      light: "#818cf8",
      dark: "#4f46e5",
    },
    secondary: {
      main: "#ec4899",
      light: "#f472b6",
      dark: "#db2777",
    },
    background: {
      default: "#0f0f23",
      paper: "#1a1a2e",
    },
    text: {
      primary: "#ffffff",
      secondary: "#a1a1aa",
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
      background: "linear-gradient(45deg, #6366f1, #ec4899)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
  },
})

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)

    try {
      console.log("Sending request to backend...")

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        mode: "cors",
        body: JSON.stringify({
          question: input,
        }),
      })

      console.log("Response status:", response.status)
      console.log("Response headers:", response.headers)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const contentType = response.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Response is not JSON")
      }

      const data = await response.json()
      console.log("Received data:", data)

      const aiMessage = {
        id: Date.now() + 1,
        type: "ai",
        content: data,
        timestamp: new Date().toLocaleTimeString(),
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error("Detailed error:", error)
      const errorMessage = {
        id: Date.now() + 1,
        type: "ai",
        content: { error: `Failed to get prediction: ${error.message}` },
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="app-container">
        <Container maxWidth="md" sx={{ height: "100vh", display: "flex", flexDirection: "column", py: 2 }}>
          {/* Header */}
          <Fade in timeout={1000}>
            <Paper
              elevation={0}
              sx={{
                p: 3,
                mb: 2,
                background: "linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1))",
                border: "1px solid rgba(99, 102, 241, 0.2)",
                borderRadius: 3,
              }}
            >
              <Box display="flex" alignItems="center" gap={2}>
                <Avatar
                  sx={{
                    background: "linear-gradient(45deg, #6366f1, #ec4899)",
                    width: 48,
                    height: 48,
                  }}
                >
                  <BrainIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="h1">
                    Bitzify Demand Forecasting
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    AI-Powered Sales Demand Prediction
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Fade>

          {/* Messages Container */}
          <Paper
            elevation={0}
            sx={{
              flex: 1,
              p: 2,
              mb: 2,
              overflow: "hidden",
              background: "rgba(26, 26, 46, 0.5)",
              border: "1px solid rgba(99, 102, 241, 0.1)",
              borderRadius: 3,
            }}
          >
            <Box
              sx={{
                height: "100%",
                overflowY: "auto",
                pr: 1,
                "&::-webkit-scrollbar": {
                  width: "6px",
                },
                "&::-webkit-scrollbar-track": {
                  background: "rgba(255, 255, 255, 0.1)",
                  borderRadius: "3px",
                },
                "&::-webkit-scrollbar-thumb": {
                  background: "linear-gradient(45deg, #6366f1, #ec4899)",
                  borderRadius: "3px",
                },
              }}
            >
              {messages.length === 0 && (
                <Fade in timeout={1500}>
                  <Box
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    justifyContent="center"
                    height="100%"
                    gap={2}
                  >
                    <TrendingUpIcon sx={{ fontSize: 64, color: "primary.main", opacity: 0.5 }} />
                    <Typography variant="h6" color="text.secondary" textAlign="center">
                      Welcome to Bitzify Demand Forecasting
                    </Typography>
                    <Typography variant="body2" color="text.secondary" textAlign="center">
                      Enter your sales data to get AI-powered demand predictions
                    </Typography>
                    <Chip
                      label="Example: S003 sold P0020 electronics in South on 2026-07-15 for 500 with 50 discount"
                      variant="outlined"
                      sx={{
                        mt: 2,
                        borderColor: "primary.main",
                        color: "primary.light",
                      }}
                    />
                  </Box>
                </Fade>
              )}

              {messages.map((message, index) => (
                <Slide key={message.id} direction="up" in timeout={500} style={{ transitionDelay: `${index * 100}ms` }}>
                  <Box mb={2}>
                    {message.type === "user" ? (
                      <Box display="flex" justifyContent="flex-end" mb={1}>
                        <Card
                          sx={{
                            maxWidth: "80%",
                            background: "linear-gradient(135deg, #6366f1, #4f46e5)",
                            borderRadius: "18px 18px 4px 18px",
                          }}
                        >
                          <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                              <Avatar sx={{ width: 24, height: 24, bgcolor: "rgba(255,255,255,0.2)" }}>
                                <PersonIcon sx={{ fontSize: 16 }} />
                              </Avatar>
                              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                You • {message.timestamp}
                              </Typography>
                            </Box>
                            <Typography variant="body1">{message.content}</Typography>
                          </CardContent>
                        </Card>
                      </Box>
                    ) : (
                      <Box display="flex" justifyContent="flex-start" mb={1}>
                        <Card
                          sx={{
                            maxWidth: "80%",
                            background: "linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(99, 102, 241, 0.1))",
                            border: "1px solid rgba(236, 72, 153, 0.2)",
                            borderRadius: "18px 18px 18px 4px",
                          }}
                        >
                          <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                              <Avatar
                                sx={{
                                  width: 24,
                                  height: 24,
                                  background: "linear-gradient(45deg, #6366f1, #ec4899)",
                                }}
                              >
                                <BrainIcon sx={{ fontSize: 16 }} />
                              </Avatar>
                              <Typography variant="caption" color="text.secondary">
                                Bitzify AI • {message.timestamp}
                              </Typography>
                            </Box>

                            {message.content.error ? (
                              <Typography variant="body1" color="error">
                                {message.content.error}
                              </Typography>
                            ) : (
                              <Box>
                                <Typography
                                  variant="h6"
                                  sx={{
                                    background: "linear-gradient(45deg, #6366f1, #ec4899)",
                                    WebkitBackgroundClip: "text",
                                    WebkitTextFillColor: "transparent",
                                    mb: 1,
                                  }}
                                >
                                  Demand Forecast Result
                                </Typography>
                                <Box
                                  sx={{
                                    p: 2,
                                    borderRadius: 2,
                                    background:
                                      "linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1))",
                                    border: "1px solid rgba(99, 102, 241, 0.3)",
                                  }}
                                >
                                  <Typography variant="body2" color="text.secondary" mb={1}>
                                    Predicted Demand Forecast:
                                  </Typography>
                                  <Typography
                                    variant="h4"
                                    sx={{
                                      fontWeight: "bold",
                                      background: "linear-gradient(45deg, #6366f1, #ec4899)",
                                      WebkitBackgroundClip: "text",
                                      WebkitTextFillColor: "transparent",
                                    }}
                                  >
                                    {message.content.predicted_demand_forecast?.toFixed(2) || "N/A"}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    units
                                  </Typography>
                                </Box>
                              </Box>
                            )}
                          </CardContent>
                        </Card>
                      </Box>
                    )}
                  </Box>
                </Slide>
              ))}

              {loading && (
                <Fade in>
                  <Box display="flex" justifyContent="flex-start" mb={1}>
                    <Card
                      sx={{
                        background: "linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(99, 102, 241, 0.1))",
                        border: "1px solid rgba(236, 72, 153, 0.2)",
                        borderRadius: "18px 18px 18px 4px",
                      }}
                    >
                      <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                        <Box display="flex" alignItems="center" gap={2}>
                          <Avatar
                            sx={{
                              width: 24,
                              height: 24,
                              background: "linear-gradient(45deg, #6366f1, #ec4899)",
                            }}
                          >
                            <BrainIcon sx={{ fontSize: 16 }} />
                          </Avatar>
                          <CircularProgress size={16} sx={{ color: "primary.main" }} />
                          <Typography variant="body2" color="text.secondary">
                            Analyzing your data...
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Box>
                </Fade>
              )}

              <div ref={messagesEndRef} />
            </Box>
          </Paper>

          {/* Input Area */}
          <Zoom in timeout={1200}>
            <Paper
              elevation={0}
              sx={{
                p: 2,
                background: "linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1))",
                border: "1px solid rgba(99, 102, 241, 0.2)",
                borderRadius: 3,
              }}
            >
              <Box display="flex" gap={1} alignItems="flex-end">
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter your sales data (e.g., S003 sold P0020 electronics in South on 2026-07-15 for 500 with 50 discount...)"
                  variant="outlined"
                  disabled={loading}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      borderRadius: 2,
                      background: "rgba(26, 26, 46, 0.5)",
                      "& fieldset": {
                        borderColor: "rgba(99, 102, 241, 0.3)",
                      },
                      "&:hover fieldset": {
                        borderColor: "rgba(99, 102, 241, 0.5)",
                      },
                      "&.Mui-focused fieldset": {
                        borderColor: "primary.main",
                      },
                    },
                  }}
                />
                <IconButton
                  onClick={handleSend}
                  disabled={!input.trim() || loading}
                  sx={{
                    background: "linear-gradient(45deg, #6366f1, #ec4899)",
                    color: "white",
                    width: 48,
                    height: 48,
                    "&:hover": {
                      background: "linear-gradient(45deg, #4f46e5, #db2777)",
                      transform: "scale(1.05)",
                    },
                    "&:disabled": {
                      background: "rgba(99, 102, 241, 0.3)",
                      color: "rgba(255, 255, 255, 0.5)",
                    },
                    transition: "all 0.2s ease-in-out",
                  }}
                >
                  <SendIcon />
                </IconButton>
              </Box>
            </Paper>
          </Zoom>
        </Container>
      </div>
    </ThemeProvider>
  )
}

export default App
