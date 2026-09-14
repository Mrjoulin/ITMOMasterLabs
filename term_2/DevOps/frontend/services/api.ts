const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api/v1';

const getAuthToken = (): string | null => {
    return localStorage.getItem('focusflow_token');
};

const setAuthToken = (token: string) => {
    localStorage.setItem('focusflow_token', token);
};

const apiFetch = async (url: string, options: RequestInit = {}) => {
    const token = getAuthToken();
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${BASE_URL}${url}`, {
        ...options,
        headers,
    });

    if (!response.ok) {
        // If unauthorized, clear token and notify app to logout
        if (response.status === 401) {
            localStorage.removeItem('focusflow_token');
            globalThis.dispatchEvent(new Event('auth:logout'));
        }
        const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred' }));
        throw new Error(errorData.detail || 'An unknown error occurred');
    }

    return response.json();
};

export const api = {
    // Auth
    login: async (formData: FormData) => {
        const response = await fetch(`${BASE_URL}/users/login`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred' }));
            throw new Error(errorData.detail || 'An unknown error occurred');
        }
        const data = await response.json();
        setAuthToken(data.access_token);
        return data;
    },

    register: async (email: string, fullName, password) => {
        const data = await apiFetch('/users/register', {
            method: 'POST',
            body: JSON.stringify({ email, full_name: fullName, password }),
        });
        setAuthToken(data.access_token);
        return data;
    },

    getUser: async () => {
        return apiFetch('/users/me');
    },

    // Areas
    getAreas: async () => {
        return apiFetch('/areas/');
    },

    createArea: async (name: string, color: string) => {
        return apiFetch('/areas/', {
            method: 'POST',
            body: JSON.stringify({ name, color }),
        });
    },

    // Tasks
    getTasks: async () => {
        return apiFetch('/tasks/');
    },

    createTask: async (task) => {
        return apiFetch('/tasks/', {
            method: 'POST',
            body: JSON.stringify(task),
        });
    },

    updateTask: async (taskId, task) => {
        const taskToUpdate = {
            title: task.title,
            description: task.description,
            due_date: task.due_date,
            completed: task.completed,
            area_id: task.area_id,
            priority: task.priority,
        };
        return apiFetch(`/tasks/${taskId}`, {
            method: 'PATCH',
            body: JSON.stringify(taskToUpdate),
        });
    },

    toggleTaskCompletion: async (taskId: string, completed: boolean) => {
        return apiFetch(`/tasks/${taskId}`, {
            method: 'PATCH',
            body: JSON.stringify({ completed: !completed }),
        });
    },

    // Notes
    getNotes: async () => {
        return apiFetch('/notes/');
    },

    createNote: async (note) => {
        return apiFetch('/notes/', {
            method: 'POST',
            body: JSON.stringify(note),
        });
    },

    updateNote: async (noteId, note) => {
        return apiFetch(`/notes/${noteId}`, {
            method: 'PATCH',
            body: JSON.stringify(note),
        });
    },

    search: async (query: string, item_type: string) => {
        if (item_type) {
            return apiFetch(`/search/?query=${encodeURIComponent(query)}&item_type=${encodeURIComponent(item_type)}`);
        }
        return apiFetch(`/search/?query=${encodeURIComponent(query)}`);
    },
};